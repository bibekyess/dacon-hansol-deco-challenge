<div style="text-align: right">
  <img src="hansolrag/tests/files/image.png" alt="Logo" align="right" width="180" />
</div>

# Hansol Deco Season 2 AI Contest: <br> Handling questions and answers about wallpaper defects

RAG Approach used to participate in Dacon Hansol Deco Challenge 2024. Please refer to [https://dacon.io/competitions/official/236216/overview/description](https://dacon.io/competitions/official/236216/overview/description) for more information about the competition.

### Installation
```shell
pip install hansolrag
```

### Inference
```shell
hansolrag --text "면진장치가 뭐야?"
hansolrag --file hansolrag/data/mini_test.csv --output-file hansolrag/deliverable/mini_test_result.json 
hansolrag --file hansolrag/data/test.csv --output-file hansolrag/deliverable/test_result.json --submission-file hansolrag/deliverable/test_result.csv
```

### Config
Please look at the [hansolrag/config/config.yaml](hansolrag/config/config.yaml) and change the config as per your preference.

In the config, you can see there are three generation-model modes:
`skt/kogpt2-base-v2` is used for quick debugging and testing purposes. The results are not satisfactory using this model.

For the best results using CPU, please download `gemma-2b-it-GGUF` from [https://huggingface.co/google/gemma-2b-it-GGUF](https://huggingface.co/google/gemma-2b-it-GGUF) and put it under this directory structure `hansolrag/model_checkpoints/gemma/gemma-2b-it.gguf`.

For the best results with GPU, we are using SOTA `OrionStarAI/Orion-14B-Chat-Int4`. Just uncomment the respective portion and run.
