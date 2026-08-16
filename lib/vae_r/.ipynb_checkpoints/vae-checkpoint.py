import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp

tfd = tfp.distributions


# ============================================================
# VAE Prob Model
# General VAE: encoder can be normal Encoder or TextEncoder
# If encoder has bert_params(), BERT gradients are scaled separately
# ============================================================
class VAE(tfk.Model):
    def __init__(self,pz,encoder,decoder,beta_kl=0.001,beta_recon=0.01,bert_lr=1e-6,name="VAEprob",**kwargs):
        super().__init__(name=name,**kwargs)

        self.pz         = pz                                         # prior distribution class
        self.encoder    = encoder                                    # Encoder or TextEncoder
        self.decoder    = decoder                                    # Decoder
        self.latent_dim = encoder.latent_dim                         # latent dimension
        self.beta_kl    = beta_kl                                    # KL weight
        self.beta_recon = beta_recon                                 # reconstruction weight
        self.bert_lr    = bert_lr                                    # BERT lr if encoder has trainable BERT

    def call(self,inputs,L=1,training=False):
        qz_x,target = self.encoder.call(
            inputs,
            training=training,
        )                                                            # q(z|x), target

        target = tf.stop_gradient(target)                            # always stop reconstruction target branch

        zs = qz_x.sample(L)                                          # L,batch,latent

        pz = self.pz(
            loc=tf.zeros(self.latent_dim),
            scale_diag=tf.ones(self.latent_dim),
        )                                                            # p(z)

        recon_sum = 0.0

        for i in range(zs.shape[0]):
            z     = zs[i,...]                                        # one latent sample
            px_z  = self.decoder(z,training=training)                # p(x|z)
            recon = px_z.log_prob(target)                            # log p(x|z)
            recon_sum += recon

        recon_mean = recon_sum/tf.cast(L,tf.float32)                 # average over L
        kl         = tfd.kl_divergence(qz_x,pz)                      # KL(q||p)

        elbo = self.beta_recon*recon_mean - self.beta_kl*kl          # weighted ELBO

        self.recon_loss = -tf.reduce_mean(recon_mean)                # unweighted recon loss for logging
        self.kl_loss    = tf.reduce_mean(kl)                         # unweighted KL for logging
        self.vae_loss   = -tf.reduce_mean(elbo)                      # weighted VAE loss
        self.loss       = self.vae_loss

        return qz_x,px_z

    @tf.function
    def train(self,x,optimizer,L=1):
        with tf.GradientTape() as tape:
            self.call(
                x,
                L=L,
                training=True,
            )

        encoder_params = self.encoder.params()                       # encoder params, may include BERT
        decoder_params = self.decoder.params()                       # decoder params

        if hasattr(self.encoder,"bert_params"):
            bert_params = self.encoder.bert_params()                 # BERT params only, empty if frozen
        else:
            bert_params = []                                         # normal Encoder has no BERT

        bert_ids         = set([id(v) for v in bert_params])
        encoder_non_bert = [v for v in encoder_params if id(v) not in bert_ids]

        params    = encoder_non_bert + decoder_params + bert_params
        gradients = tape.gradient(self.loss,params)

        n_enc = len(encoder_non_bert)
        n_dec = len(decoder_params)

        encoder_gradients = gradients[:n_enc]
        decoder_gradients = gradients[n_enc:n_enc+n_dec]
        bert_gradients    = gradients[n_enc+n_dec:]

        if len(bert_params)>0:
            base_lr    = tf.cast(optimizer.learning_rate,tf.float32)
            bert_scale = tf.cast(self.bert_lr,tf.float32)/base_lr

            bert_gradients = [
                None if g is None else g*bert_scale
                for g in bert_gradients
            ]                                                       # effective BERT lr = bert_lr

        final_gradients = encoder_gradients + decoder_gradients + bert_gradients

        optimizer.apply_gradients([
            (g,v) for g,v in zip(final_gradients,params) if g is not None
        ])

        return self.loss
