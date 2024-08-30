# Running our method with `execute.py`

To execute, configure the following parameters:

1. **`stage`**: `int`
   - Indicates the stage of the selection process, 1 or 2.

2. **`detail`**: `Boolean` or `string`
   - Specifies the level of detail to include:
   - - `False` for no detail
     - `True` for all details
     - `'Education'` for education details only
     - `'Work'` for work details only

3. **`model`**: `str` or others
   - an object string for llm models such as `'deepseek-chat'`
   - or other traditional machine learning models such as `XGBClassifier()`

4. **`save_result`**: `Boolean`
    - whether the prediction result is saved, if True, the result will be saved under Result_Round{stage}

---
### The following parameters only apply to LLM methods:
1. **`prompt_type`**: `string`
   - Specifies the type of prompt to use. Options include:
     - `'None'`: Default setting, only contains basic context and test data
     - `'Train'`: Add respective training samples when applicable to default
     - `'Summary'`: Add llm summary generated using respective training samples when applicable to default
     - `'Train+Summary'`: Add llm summary and respective training samples when applicable to default
     - `'Ruleset'`: Add ruleset generated using respective training samples when applicable to default
     - `'Prototype'`: Add prototype generated using respective training samples when applicable to default

2. **`shuffle`**: `Boolean`
   - Determines whether the data should be shuffled.

3. **`vote_count`**: `int`
   - Specifies the number of LLM agents that will vote.

4. **`n_retry`**: `int`
   - Sets the maximum number of retries for the LLM response to conform to the designated format.

5. **`do_small_group`**: `Boolean`
   - Indicates whether the data should be processed in smaller groups repeatedly(i.e. only twice the size of the desired candidate size is used each time), this will only apply to those tasks where the number of candidates exceed `sg_threshold`, which default is set to 30.

6. **`client`**: `OpenAI object`
   - Your OpenAI API key and base URL.