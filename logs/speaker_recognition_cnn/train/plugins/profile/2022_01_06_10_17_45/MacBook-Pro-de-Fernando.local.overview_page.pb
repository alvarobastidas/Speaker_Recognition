?	???K?V?@???K?V?@!???K?V?@	P?NGtp?P?NGtp?!P?NGtp?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???K?V?@?t??@A??~j?M?@Y+??????*	    PA2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip[0]::Map??~j?te@!		?X@)?G?ze@1??o??oX@:Preprocessing2?
`Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::ShuffleFףp=
?_@!@/@/?R@)D?l?????1;?;???:Preprocessing2?
eIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip~?rh??e@!????X@)??S㥛??1??????:Preprocessing2?
zIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip[0]::Map::TensorSlice??K7?A`??!??~??~??)?K7?A`??1??~??~??:Preprocessing2?
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip[1]::TensorSliceF????x???!b?b???)????x???1b?b???:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2{?G?z??!???ͧ?){?G?z??1???ͧ?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???x?&??!??????)???x?&??1??????:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2?p=
ף??!,W,W??)?p=
ף??1,W,W??:Preprocessing2F
Iterator::Model
ףp=
??!x?xǪ?)9??v????1????~?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??ʡE???!<?<???){?G?z??1????w?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9P?NGtp?I'??.??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?t??@?t??@!?t??@      ??!       "      ??!       *      ??!       2	??~j?M?@??~j?M?@!??~j?M?@:      ??!       B      ??!       J	+??????+??????!+??????R      ??!       Z	+??????+??????!+??????b      ??!       JCPU_ONLYYP?NGtp?b q'??.??X@Y      Y@q?q??%V?"?
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