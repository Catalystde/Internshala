---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:15029
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: quotes about humor
  sentences:
  - '“Sometimes when I''m talking, my words can''t keep up with my thoughts. I wonder
    why we think faster than we speak. Probably so we can think twice.” by Bill Watterson.
    Tags: calvin-and-hobbes, humor'
  - '“Lost opportunities, lost possibilities, feelings we can never get back. That''s
    part of what it means to be alive. But inside our heads - at least that''s where
    I imagine it - there''s a little room where we store those memories. A room like
    the stacks in this library. And to understand the workings of our own heart we
    have to keep on making new reference cards. We have to dust things off every once
    in awhile, let in fresh air, change the water in the flower vases. In other words,
    you''ll live forever in your own private library.” by Haruki Murakami,. Tags:
    inner-life, life, missed-chances'
  - '“A cynic is a man who knows the price of everything, and the value of nothing.”
    by Oscar Wilde. Tags: cynicism'
- source_sentence: quotes about world
  sentences:
  - '“There is only one thing that makes a dream impossible to achieve: the fear of
    failure.” by Paulo Coelho,. Tags: achievement, dreams, failure, fear'
  - '“It doesn''t interest me what you do for a living. I want to know what you ache
    for, and if you dare to dream of meeting your heart''s longing.It doesn''t interest
    me how old you are. I want to know if you will risk looking like a fool for love,
    for your dream, for the adventure of being alive. It doesn''t interest me what
    planets are squaring your moon. I want to know if you have touched the center
    of your own sorrow, if you have been opened by life''s betrayals or have become
    shriveled and closed from fear of further pain!I want to know if you can sit with
    pain, mine or your own, without moving to hide it or fade it, or fix it. I want
    to know if you can be with joy, mine or your own, if you can dance with wildness
    and let the ecstasy fill you to the tips of your fingers and toes without cautioning
    us to be careful, to be realistic, to remember the limitations of being human.
    It doesn''t interest me if the story you are telling me is true. I want to know
    if you can disappoint another to be true to yourself; if you can bear the accusation
    of betrayal and not betray your own soul; if you can be faithlessand therefore
    trustworthy. I want to know if you can see beauty even when it''s not pretty,
    every day,and if you can source your own life from its presence. I want to know
    if you can live with failure, yours and mine, and still stand on the edge of the
    lake and shout to the silver of the full moon, "Yes!"�It doesn''t interest me
    to know where you live or how much money you have. I want to know if you can get
    up, after the night of grief and despair, weary and bruised to the bone, and do
    what needs to be done to feed the children. It doesn''t interest me who you know
    or how you came to be here. I want to know if you will stand in the center of
    the fire with me and not shrink back. It doesn''t interest me where or what or
    with whom you have studied. I want to know what sustains you, from the inside,
    when all else falls away. I want to know if you can be alone with yourself and
    if you truly like the company you keep in the empty moments.” by Oriah Mountain
    Dreamer. Tags: inspirational'
  - '“Don''t go around saying the world owes you a living. The world owes you nothing.
    It was here first.” by Mark Twain. Tags: living, world'
- source_sentence: quotes about inspirational
  sentences:
  - '“there are two types of people in the world: those who prefer to be sad among
    others, and those who prefer to be sad alone.” by Nicole Krauss,. Tags: comfort,
    sad, sadness'
  - '“Why did you do all this for me?'' he asked. ''I don''t deserve it. I''ve never
    done anything for you.'' ''You have been my friend,'' replied Charlotte. ''That
    in itself is a tremendous thing.” by E.B. White,. Tags: friendship'
  - '“I can''t give you a sure-fire formula for success, but I can give you a formula
    for failure: try to please everybody all the time.” by Herbert Bayard Swope. Tags:
    failure, inspirational, misattributed-bill-cosby, success'
- source_sentence: weather forecast
  sentences:
  - '“The way to get started is to quit talking and begin doing. ” by Walt Disney.
    Tags: motivation, success'
  - '“Usually we walk around constantly believing ourselves. "I''m okay" we say. "I''m
    alright". But sometimes the truth arrives on you and you can''t get it off. That''s
    when you realize that sometimes it isn''t even an answer--it''s a question. Even
    now, I wonder how much of my life is convinced.” by Markus Zusak,. Tags: truth'
  - '“You know you''re in love when you can''t fall asleep because reality is finally
    better than your dreams.” by Dr. Seuss. Tags: dreams, love, reality, sleep'
- source_sentence: hamlet quotes
  sentences:
  - '“The reason I talk to myself is because I''m the only one whose answers I accept.”
    by George Carlin. Tags: humor, insanity, lies, lying, self-indulgence, truth'
  - '“Sometimes you lose a battle. But mischief always wins the war” by John Green,.
    Tags: alaska-young'
  - '“We know what we are, but not what we may be.” by William Shakespeare. Tags:
    hamlet, identity, possibilities'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'hamlet quotes',
    '“We know what we are, but not what we may be.” by William Shakespeare. Tags: hamlet, identity, possibilities',
    "“The reason I talk to myself is because I'm the only one whose answers I accept.” by George Carlin. Tags: humor, insanity, lies, lying, self-indulgence, truth",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 15,029 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                      | sentence_1                                                                          | label                                                          |
  |:--------|:--------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                          | string                                                                              | float                                                          |
  | details | <ul><li>min: 4 tokens</li><li>mean: 6.4 tokens</li><li>max: 19 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 55.74 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.79</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                              | sentence_1                                                                                                                                                                                              | label            |
  |:----------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>you can't have love”</code>       | <code>“sex is the consolation you have when you can't have love” by Gabriel GarcÃ­a MÃ¡rquez. Tags: desire, loneliness, love, lust, passion, sex</code>                                                 | <code>1.0</code> |
  | <code>weather forecast</code>           | <code>“The greatest enemy of knowledge is not ignorance, it is the illusion of knowledge.” by Daniel J. Boorstin. Tags: ignorance, knowledge, misattributed-stephen-hawking</code>                      | <code>0.0</code> |
  | <code>sayings by Albert Einstein</code> | <code>“If you want your children to be intelligent, read them fairy tales. If you want them to be more intelligent, read them more fairy tales.” by Albert Einstein. Tags: children, fairy-tales</code> | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.5319 | 500  | 0.0887        |
| 1.0638 | 1000 | 0.0486        |
| 1.5957 | 1500 | 0.0429        |
| 2.1277 | 2000 | 0.0404        |
| 2.6596 | 2500 | 0.0377        |


### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 4.1.0
- Transformers: 4.53.0
- PyTorch: 2.6.0+cu124
- Accelerate: 1.8.1
- Datasets: 2.14.4
- Tokenizers: 0.21.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->