import inference_modified  # from POINTER

inference_modified.main([
	'--bert_model', '../POINTER/ckpt/assistant_model_maison/checkpoint-20000',
	'--type', 'greedy',
	'--no_ins_at_end', 'when_punctuation_at_end',
	'--keyfile', 'tmp_in.test.txt',
	'--output_file', 'tmp_out.test.txt'
])