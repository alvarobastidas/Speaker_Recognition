?	H?znϧ@H?znϧ@!H?znϧ@	F?[Lm?F?[Lm?!F?[Lm?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$H?znϧ@??K7?A??AV-?ͧ@Y??C?l???*	    ???@2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip[0]::Map?-????Z@!?6?&??X@)%??CZ@1?5?gԆX@:Preprocessing2?
eIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip?q=
ף`Z@!k
???X@)??~j?t??1"??????:Preprocessing2?
zIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip[0]::Map::TensorSlice?1?Zd??!w? ????)1?Zd??1w? ????:Preprocessing2?
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip[1]::TensorSlice:?O??n??!_S?n????)?O??n??1_S?n????:Preprocessing2?
`Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle:;?O??.L@!Oj?n?J@)??Q???1y~????:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2P??n???!?*'?+#??)P??n???1?*'?+#??:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2?ʡE????!Xz)??=??)?ʡE????1Xz)??=??:Preprocessing2F
Iterator::Model333333??!???f??)333333??1???f??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchT㥛? ??!Ծgq?i??)T㥛? ??1Ծgq?i??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9G?[Lm?I?)Hg??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??K7?A????K7?A??!??K7?A??      ??!       "      ??!       *      ??!       2	V-?ͧ@V-?ͧ@!V-?ͧ@:      ??!       B      ??!       J	??C?l?????C?l???!??C?l???R      ??!       Z	??C?l?????C?l???!??C?l???b      ??!       JCPU_ONLYYG?[Lm?b q?)Hg??X@Y      Y@q??'?Z?"?
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