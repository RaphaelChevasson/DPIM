# Code

Coming soon! Code is being cleaned and tested against fresh windows and linux right now.

`/!\` Sorry for the delay, we got hit pretty hard by COVID. Still comming soon `/!\`

# Computational requirements

There are 4 steps with a significant runtime:
- POINTER fine-tuning: This step is batched and scalable to multiple GPUs. We found fine-tuning for 9 gpu-hour per dataset (around 10 epochs) to be enough for our use case.
- Synonym extraction: Without batching, we generate all contextual synonyms of 3 source sentence per gpu-second, making a run on the biggest dataset about 10 gpu-hours. We amortize it largely by caching it between the runs.
- POINTER generation: Without batching, we generate 2.6 paraphrases per gpu-seconds, making a run on the biggest dataset about 12 gpu-hours.
- Paraphrase evaluation: For the fluency evaluation, we use a distilgpt2 model. A run on the biggest dataset takes about 2 gpu-hour.

Memory-wise, all steps have been found to fit on a consumer-grade 8Go Nvidia gtx1070, although larger cards are recommended for larger batch size in the fine-tuning step.
