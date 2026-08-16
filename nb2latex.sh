#!/bin/bash

INPUT_DIRS=(
    "nb/data_preprocessing"
    "nb/feature_analysis"
)

RUN_DATE_TIME="$(date +"%d_%H%M%S")"

OUTPUT_ROOT="latex"
OUTPUT_DIR="${OUTPUT_ROOT}_${RUN_DATE_TIME}"

# Create fresh output folder
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

for INPUT_DIR in "${INPUT_DIRS[@]}"; do

    echo "Processing input folder: $INPUT_DIR"

    # Skip if folder does not exist
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Skipping missing folder: $INPUT_DIR"
        continue
    fi

    # Remove Jupyter checkpoint folders before conversion
    find "$INPUT_DIR" -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

    find "$INPUT_DIR" \
        -type d -name ".ipynb_checkpoints" -prune -o \
        -type f -name "*.ipynb" -print0 | while IFS= read -r -d '' ipynb_file; do

        # Keep folder structure after nb/
        rel_path="${ipynb_file#nb/}"
        rel_no_ext="${rel_path%.ipynb}"

        out_dir="$OUTPUT_DIR/$(dirname "$rel_no_ext")"
        tex_file="$OUTPUT_DIR/${rel_no_ext}.tex"

        mkdir -p "$out_dir"

        tmp_py="$(mktemp).py"

        # Convert notebook to python script text
        jupyter nbconvert \
            --to script "$ipynb_file" \
            --stdout > "$tmp_py"

        # Remove notebook cell marker prefix like:
        # # In[30]:
        # # In[30]: some code
        perl -0pi -e 's/^# In\[[^\]]*\]:[ \t]*//mg' "$tmp_py"

        # Get parent folder and file name
        parent_folder="$(basename "$(dirname "$ipynb_file")")"
        file_name="$(basename "$rel_no_ext")"

        # Create label as parent_folder_file_name
        label_slug="${parent_folder}_${file_name}"

        # Make label safe for LaTeX
        label_slug=$(echo "$label_slug" \
            | tr '[:upper:]' '[:lower:]' \
            | sed 's#[/ .-]#_#g' \
            | sed 's/[^a-z0-9_]//g' \
            | sed 's/_\+/_/g' \
            | sed 's/^_//;s/_$//')

        listing_label="lst:${label_slug}"

        # Caption text, escape underscores
        caption_text=$(basename "$rel_no_ext" | sed 's/_/\\_/g')

        # Create LaTeX file
        {
            echo "% Auto-generated from: $ipynb_file"
            echo "% Listing label: $listing_label"
            echo "\\begin{lstlisting}[style=python,caption={$caption_text},label={$listing_label}]"
            cat "$tmp_py"
            echo "\\end{lstlisting}"
        } > "$tex_file"

        rm "$tmp_py"

        echo "Created: $tex_file"
        echo "Label:   $listing_label"
    done

done

echo "Done. LaTeX files written to: $OUTPUT_DIR"