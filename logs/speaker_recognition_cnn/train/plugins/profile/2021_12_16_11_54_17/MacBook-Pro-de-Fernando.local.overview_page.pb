?	+?V??@+?V??@!+?V??@	?t??m??t??m?!?t??m?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$+?V??@??v????A?|?5??@YX9??v???*	    ?'?@2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip[0]::Map?fffffFZ@!
????X@)X9??v&Z@1*Z??hX@:Preprocessing2?
eIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip?X9?ȆZ@!?Nr??X@)???(\???1t5?????:Preprocessing2?
zIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip[0]::Map::TensorSlice?V-?????!??x&Y???)V-?????1??x&Y???:Preprocessing2?
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip[1]::TensorSlice6q=
ףp??!?	%?z??)q=
ףp??1?	%?z??:Preprocessing2?
`Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle5? ?rhAG@!h<???E@)??ʡE???1?0:?Tf??:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2Zd;?O???!x??????)Zd;?O???1x??????:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2?p=
ף??!mzׁ??)?p=
ף??1mzׁ??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch9??v????!#??xڨ?)9??v????1#??xڨ?:Preprocessing2F
Iterator::Model#??~j???!?,?[??){?G?z??1̭g????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Zd;??!???C'??);?O??n??1?C,w4??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?t??m?I??????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??v??????v????!??v????      ??!       "      ??!       *      ??!       2	?|?5??@?|?5??@!?|?5??@:      ??!       B      ??!       J	X9??v???X9??v???!X9??v???R      ??!       Z	X9??v???X9??v???!X9??v???b      ??!       JCPU_ONLYY?t??m?b q??????X@Y      Y@qq
i{+R?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 