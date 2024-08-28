Please use run_llm to run the model, the parameters are:
1. 'prompt_type': string; this can either be 'None', 'Train', 'Summary', 'Train+Summary', 'Ruleset', 'Prototype'
2. 'stage': int;  1/2 referring to round 1 or 2
3. 'shuffle': Boolean; whether or not the data is shuffled
4. 'detail': Boolean or string; True means using all detail, 'Education' means only use education details, 'Work' means only use work details
5. 'vote_count': int; the number of llm agents that votes
6. 'n_retry': int; athe maximun number of retries for the llm response to give the designated format
7. 'do_small_group': Boolean; whether or not the data is given in smaller groups repeatedly
8. 'client': OpenAI object; your openai api key and base url
9. 'model': object; llm model used to get responses
