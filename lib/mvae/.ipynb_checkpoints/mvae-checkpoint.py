import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as tfk
tfd = tfp.distributions


# ============================================================
# Modality as Keras Model
# ============================================================
class Modality(tfk.Model):
    def __init__(self,name,encoder,decoder,**kwargs):
        super().__init__(name=name,**kwargs)
        self.encoder = encoder                                      # Encoder or TextEncoder
        self.decoder = decoder                                      # Decoder

    @property
    def numel(self):
        return self.encoder.input_dim                               # reconstruction dimension

    @property
    def latent_dim(self):
        return self.encoder.latent_dim                              # latent dimension

    def params(self):
        return self.encoder.params() + self.decoder.params()         # encoder + decoder params

    def qz_x(self,x,training=False):
        return self.encoder(x,training=training)                    # q(z|x), target

    def px_z(self,z,training=False):
        return self.decoder(z,training=training)                    # p(x|z)

    def call(self,x,training=False):
        return self.qz_x(x,training=training)                       # keras tracking


# ============================================================
# MVAE
# m1 = EDA/PPG
# m2 = PPG/TEXT
# TEXT only in m2
# ============================================================
class MVAE(tfk.Model):
    def __init__(self,pz,m1_modality,m2_modality,beta_kl=0.001,beta_recon=0.01,
                 fusion_method="moe",moe_weighting="equal",bert_lr=1e-6,
                 name="MVAE",**kwargs):
        super().__init__(name=name,**kwargs)

        self.pz            = pz                                      # prior distribution class
        self.beta_kl       = beta_kl                                 # KL weight
        self.beta_recon    = beta_recon                              # reconstruction weight
        self.fusion_method = fusion_method                           # moe / poe
        self.moe_weighting = moe_weighting                           # equal / dimension
        self.bert_lr       = bert_lr                                 # BERT lr if text encoder trains BERT

        self.m1 = m1_modality                                       # EDA/PPG
        self.m2 = m2_modality                                       # PPG/TEXT

        if self.fusion_method not in ["moe","poe"]:
            raise ValueError("fusion_method must be either 'moe' or 'poe'.")

        if self.moe_weighting not in ["equal","dimension"]:
            raise ValueError("moe_weighting must be either 'equal' or 'dimension'.")

        if self.m1.latent_dim != self.m2.latent_dim:
            raise ValueError("Both modalities must have same latent_dim.")

        self.latent_dim = self.m1.latent_dim                         # shared latent dimension

    def get_effective_alpha(self):
        if self.moe_weighting=="equal":
            alpha = tf.constant([0.5,0.5],dtype=tf.float32)          # equal MoE weight

        if self.moe_weighting=="dimension":
            m1_numel = tf.cast(self.m1.numel,tf.float32)
            m2_numel = tf.cast(self.m2.numel,tf.float32)
            total    = m1_numel + m2_numel
            alpha    = tf.stack([m1_numel/total,m2_numel/total],axis=0)

        alpha = tf.reshape(alpha,(2,1,1))                            # broadcast over batch and latent dim

        return alpha

    def fuse_moe(self,qz_m1,qz_m2):
        mu_stack  = tf.stack([qz_m1.mean(),qz_m2.mean()],axis=0)     # 2,batch,latent
        var_stack = tf.stack([qz_m1.variance(),qz_m2.variance()],axis=0)

        var_stack = tf.maximum(var_stack,1e-6)                       # numerical safety
        alpha     = self.get_effective_alpha()                       # modality weights

        fused_mu = tf.reduce_sum(alpha*mu_stack,axis=0)              # weighted mean

        second_moment = tf.reduce_sum(
            alpha*(var_stack + tf.square(mu_stack)),
            axis=0,
        )                                                           # E[z^2]

        fused_var = second_moment - tf.square(fused_mu)              # Var[z]
        fused_var = tf.maximum(fused_var,1e-6)

        qz_fused = tfd.MultivariateNormalDiag(
            loc=fused_mu,
            scale_diag=tf.sqrt(fused_var) + 1e-6,
        )

        return qz_fused

    def fuse_poe(self,qz_m1,qz_m2):
        mu_stack  = tf.stack([qz_m1.mean(),qz_m2.mean()],axis=0)     # expert means
        var_stack = tf.stack([qz_m1.variance(),qz_m2.variance()],axis=0)

        var_stack       = tf.maximum(var_stack,1e-6)
        precision_stack = 1.0/var_stack                              # expert precision

        prior_mu        = tf.zeros_like(mu_stack[0])                 # prior mean
        prior_var       = tf.ones_like(var_stack[0])                 # prior variance
        prior_precision = 1.0/prior_var                              # prior precision

        fused_precision = prior_precision + tf.reduce_sum(precision_stack,axis=0)
        fused_var       = 1.0/fused_precision

        fused_mu = fused_var*(
            prior_precision*prior_mu +
            tf.reduce_sum(precision_stack*mu_stack,axis=0)
        )

        qz_fused = tfd.MultivariateNormalDiag(
            loc=fused_mu,
            scale_diag=tf.sqrt(fused_var) + 1e-6,
        )

        return qz_fused

    def fuse(self,qz_m1,qz_m2):
        if self.fusion_method=="moe":
            return self.fuse_moe(qz_m1,qz_m2)                       # MoE fusion

        if self.fusion_method=="poe":
            return self.fuse_poe(qz_m1,qz_m2)                       # PoE fusion

    def encode_m1(self,x,training=False):
        if not isinstance(x,dict):
            x = tf.cast(x,tf.float32)                                # physiological input

        qz_x,target = self.m1.qz_x(x,training=training)

        return qz_x,target

    def encode_m2(self,x,training=False):
        if not isinstance(x,dict):
            x = tf.cast(x,tf.float32)                                # PPG input

        qz_x,target = self.m2.qz_x(x,training=training)              # PPG or TEXT

        return qz_x,target

    def get_posteriors(self,inputs,training=False):
        x_m1 = inputs["m1"]
        x_m2 = inputs["m2"]

        qz_m1,target_m1 = self.encode_m1(x_m1,training=training)
        qz_m2,target_m2 = self.encode_m2(x_m2,training=training)

        qz_fused = self.fuse(qz_m1,qz_m2)                            # fused posterior

        target_dict = {
            "m1":target_m1,
            "m2":target_m2,
        }

        return qz_fused,[qz_m1,qz_m2],target_dict

    def call(self,inputs,L=1,training=False):
        qz_fused,qz_list,target_dict = self.get_posteriors(inputs,training=training)

        pz = self.pz(
            loc=tf.zeros(self.latent_dim),
            scale_diag=tf.ones(self.latent_dim),
        )                                                           # p(z)

        recon = 0.0

        for l in range(L):
            z = qz_fused.sample()                                    # latent sample

            px_m1 = self.m1.px_z(z,training=training)
            px_m2 = self.m2.px_z(z,training=training)

            recon_m1 = px_m1.log_prob(target_dict["m1"]) / tf.cast(self.m1.numel,tf.float32)  # dim-normalized
            recon_m2 = px_m2.log_prob(target_dict["m2"]) / tf.cast(self.m2.numel,tf.float32)  # dim-normalized

            recon += (recon_m1 + recon_m2)/2.0                       # average modalities

        recon_mean = recon/tf.cast(L,tf.float32)

        kl   = tfd.kl_divergence(qz_fused,pz)                        # KL(q||p)
        elbo = self.beta_recon*recon_mean - self.beta_kl*kl          # ELBO

        self.recon_loss = -tf.reduce_mean(recon_mean)
        self.kl_loss    = tf.reduce_mean(kl)
        self.vae_loss   = -tf.reduce_mean(elbo)
        self.loss       = self.vae_loss

        return qz_fused

    @tf.function
    def train(self,inputs,optimizer,L=1):
        with tf.GradientTape() as tape:
            self.call(inputs,L=L,training=True)

        all_params = self.m1.params() + self.m2.params()             # all trainable params

        if hasattr(self.m2.encoder,"bert_params"):
            bert_params = self.m2.encoder.bert_params()                     # only TextEncoder has this
        else:
            bert_params = []                                                 # normal Encoder has no BERT                                    # no BERT for PPG

        bert_ids        = set([id(v) for v in bert_params])
        non_bert_params = [v for v in all_params if id(v) not in bert_ids]

        params    = non_bert_params + bert_params
        gradients = tape.gradient(self.loss,params)

        non_bert_gradients = gradients[:len(non_bert_params)]
        bert_gradients     = gradients[len(non_bert_params):]

        if len(bert_params)>0:
            base_lr    = tf.cast(optimizer.learning_rate,tf.float32) # current optimizer lr
            bert_scale = tf.cast(self.bert_lr,tf.float32)/base_lr    # scale BERT gradients

            bert_gradients = [
                None if g is None else g*bert_scale
                for g in bert_gradients
            ]                                                       # BERT effective lr = bert_lr

        final_gradients = non_bert_gradients + bert_gradients

        optimizer.apply_gradients([
            (g,v) for g,v in zip(final_gradients,params) if g is not None
        ])

        return self.loss

    def get_alpha_values(self):
        if self.fusion_method=="moe":
            if self.moe_weighting=="equal":
                return {"m1":0.5,"m2":0.5}

            if self.moe_weighting=="dimension":
                m1_numel = float(self.m1.numel)
                m2_numel = float(self.m2.numel)
                total    = m1_numel + m2_numel

                return {"m1":m1_numel/total,"m2":m2_numel/total}

        if self.fusion_method=="poe":
            return {"m1":"precision-weighted","m2":"precision-weighted"}
