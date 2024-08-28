# Running the Model with `run_llm`

To execute the model using the `run_llm` function, configure the following parameters:

1. **`prompt_type`**: `string`
   - Specifies the type of prompt to use. Options include:
     - `'None'`: Default setting, only contains basic context and test data
     - `'Train'`: Add respective training samples when applicable to default
     - `'Summary'`: Add llm summary generated using respective training samples when applicable to default
     - `'Train+Summary'`: Add llm summary and respective training samples when applicable to default
     - `'Ruleset'`: Add ruleset generated using respective training samples when applicable to default
     - `'Prototype'`: Add prototype generated using respective training samples when applicable to default

2. **`stage`**: `int`
   - Indicates the stage of the process. Use `1` for round 1 or `2` for round 2.

3. **`shuffle`**: `Boolean`
   - Determines whether the data should be shuffled.

4. **`detail`**: `Boolean` or `string`
   - Specifies the level of detail to include:
   - - `False` for no detail
     - `True` for all details
     - `'Education'` for education details only
     - `'Work'` for work details only

5. **`vote_count`**: `int`
   - Specifies the number of LLM agents that will vote.

6. **`n_retry`**: `int`
   - Sets the maximum number of retries for the LLM response to conform to the designated format.

7. **`do_small_group`**: `Boolean`
   - Indicates whether the data should be processed in smaller groups repeatedly.

8. **`client`**: `OpenAI object`
   - Your OpenAI API key and base URL.

9. **`model`**: `object`
   - The LLM model used to generate responses.
