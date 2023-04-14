# Installation

```sh
# clone this repo
git clone https://github.com/RaphaelChevasson/DPIM.git
# and his dependencies
git submodule init
git submodule update

# setup python environments, datasets and models
./setup.sh
```

# Folder structure

Directories:
```
repo root   # License, citing info and aknowledgments
 |- result  # Paraphrase samples and metrics
 |- code    # Code for NAR paraphrasing, evaluation and fine-tuning
     |- paraphrase  # code for our NAR paraphrasing framework
     |- ewiser      # dependency for contextual synonym augmentation
     |- POINTER     # dependency for NAR sentence expansion
     |- aeda        # dependency for paraphrase generation of EDA and AEDA baselines
     |- ProtAugment # dependency for paraphrase generation of bart-uni baseline, datasets,
                    #  and evaluation through the ProtAugment meta-learning framework
```

Files:
```
repo root
 |- result
 |- code
     |- paraphrase
         |- augmentation_pipeline.py  # main script to run our framework
         |- run.sh                    # schedule several experiments on a distributed cluster
         |- evaluate_paraphrase_diversity.py  # main script to compute metrics, make plots and tables
     |- ewiser
         |- eval_wsd.py               # main script
     |- POINTER
         |- inference_modified.py     # inference script, modified to enable preventing insertion at the end of
                                      #  the sentence, which is used when we keep the punctuation marks '?' '!'
         |- training_modified.py      # training script, modified to make a checkpoint every power of two epochs
     |- aeda
         |- code/                     # main scripts
         |- experiments/_*.{sh,py}    # our attempt to reproduce all results on (A)EDA datasets.
                                      # order is `_setup_xps.sh`, then `_run*.sh`, then `_post_xps.sh`
     |- ProtAugment
         |- models/proto/protaugment_modified.py  # inference script, modified to fix a bug where tensorflow
                                                  #  took all GPU memory making pytorch error out
```

# Computational requirements

There are 4 steps with a significant runtime:
- POINTER fine-tuning: This step is batched and scalable to multiple GPUs. We found fine-tuning for 9 gpu-hour per dataset (around 10 epochs) to be enough for our use case.
- Synonym extraction: Without batching, we generate all contextual synonyms of 3 source sentence per gpu-second, making a run on the biggest dataset about 10 gpu-hours. We amortize it largely by caching it between the runs.
- POINTER generation: Without batching, we generate 2.6 paraphrases per gpu-seconds, making a run on the biggest dataset about 12 gpu-hours.
- Paraphrase evaluation: For the fluency evaluation, we use a distilgpt2 model. A run on the biggest dataset takes about 2 gpu-hour.

Memory-wise, all steps have been found to fit on a consumer-grade 8Go Nvidia gtx1070, although larger cards are recommended for larger batch size in the fine-tuning step.

# Running the paraphrasing pipeline

```sh
cd code/paraphrase
source ../ewiser/venv-ewiser/bin/activate

dataset=BANKING77
xp_dir=xps/001-$dataset/
mkdir $xp_dir
python augmentation_pipeline.py True True True  # 3 booleans describing which part of the pipeline to launch:
                                                #  - do_paraphrase to extract and augment keywords,
                                                #  - do_pointer to generate paraphrases from them,
                                                #  - do_protaugment to evaluate paraphrases using the ProtAugment framework.
                                                #  intermediate result are kept in --tmp_path so those can be made in multiple runs.
    $dataset # dataset to use (in BANKING77, HWU64, OOS, Liu).
    --limit_dataset 0  # do we truncate the dataset for a faster, less acurate run
                       #  0: do not truncate ; 1000: keep only 1000 source sentences
    --p_synonyms 0.75 --p_shuffles 0.25  # keyword augmentation probabilities
    --pointer_model ../POINTER/ckpt/assistant_model  # expension model to use
    --n_outputs 5  # number of paraphrases per sentence
    --regimes full --K 1 --CV 1  # ProtAugment evaluation parametters
    --log_path $xp_dir --tmp_path $xp_dir/tmp --out_path $xp_dir  # output directories
```

Alternatively, you can use your own source sentences by:
 - entering any name in place of `$dataset`
 - setting `--in_file` to your text file containing one sample per line
 - setting `do_protaugment` to `False` (if you want to evaluate with ProtAugment, you can set it to `True` but you will need to provide your labels and samples in a similar form to the 4 existing datasets.)
for example:
```sh
xp_dir=xps/002-custom_samples/
mkdir $xp_dir
python augmentation_pipeline.py True True False  # do_paraphrase, do_pointer, do_protaugment
    custom_samples --in_file my_samples.txt  # use custom source sentences
    --p_synonyms 0.75 --p_shuffles 0.25  # keyword augmentation probabilities
    --pointer_model ../POINTER/ckpt/assistant_model  # expension model to use
    --n_outputs 5  # number of paraphrases per sentence
    --log_path $xp_dir --tmp_path $xp_dir/tmp --out_path $xp_dir  # output directories
```
    
If you plan to run multiple experiments on a slurm-managed cluster, you can find a script to do so in `code/paraphrase/run.sh`. Edit it according to your needs and run it on the slurm gateway to schedule jobs in multiple nodes.


# Fine-tuning the generation model (POINTER) on your own dataset

If you want to finetune from their public wikipedia checkpoint like we did, you might want to download it from [here](https://yizzhang.blob.core.windows.net/insertiont/ckpt/wiki.tar.gz?sv=2019-10-10&st=2021-03-10T21%3A40%3A57Z&se=2030-03-11T20%3A40%3A00Z&sr=b&sp=r&sig=oYI%2BKrj5wqeFV5jAF6EY15P8%2BtNGI%2F7FIOEox08QFDY%3D) and unizip it in `code/POINTER/ckpt/wiki_model`

```sh
cd code/POINTER
source venv-POINTER/bin/activate

# build the progressive generation dataset by converting each sentence into multiple generation steps:
dataset_in=./data/all_train.txt  # must be a text file with one sample per line
dataset_out=./data/all_train_processed/  # will be created and filled with the training episodes
python generate_training_data.py --train_corpus $dataset_in --bert_model bert-base-uncased --output_dir $dataset_out --clean --task_name my_task

# finetune on a single GPU, from POINTER wikipedia checkpoint:
model_in=./ckpt/wiki_model
model_out=./ckpt/my_model
python training.py --pregenerated_data $dataset_out --bert_model $model_in --output_dir $model_out --epochs 40 --train_batch_size 1 --output_step 1000 --learning_rate 1e-5

# finetune on multiple GPUs, from POINTER wikipedia checkpoint:
num_gpus=4
model_in=./ckpt/wiki_model
model_out=./ckpt/my_model
python -m torch.distributed.launch  --nproc_per_node $num_gpus training.py --pregenerated_data $dataset_out --bert_model $model_in --output_dir $model_out --epochs 40 --train_batch_size 16 --output_step 20000 --learning_rate 1e-5
```

We recommand adjusting `--train_batch_size` to the max that don't get you a cuda (GPU) out of memory error.  
With big datasets, you might want to add `--reduce_memory` to load training data as on-disc memmaps if you get a non-cuda (CPU) out of memory error.  
See POINTER repo's [documentation](https://github.com/dreasysnail/POINTER) for more details on fine-tuning POINTER or training it from scratch.  
