?	?|?5??@?|?5??@!?|?5??@	h?㢩?b?h?㢩?b?!h?㢩?b?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?|?5??@=
ףp=??A+?????@Y??(\?µ?*	    AA2?
^Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Zip[0]::Map?????Mvd@!,??6?X@)D?l??ad@1V?:???X@:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Zip???Q??d@!?F-e?X@)?"??~j??1Ƴ{?????:Preprocessing2?
kIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Zip[0]::Map::TensorSlice?R???Q??!??Y?[~??)R???Q??1??Y?[~??:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Zip[1]::TensorSliceP/?$????!=K#>???)/?$????1=K#>???:Preprocessing2?
QIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::ShuffleP9??v?W^@!?g???IR@)'1?Z??1$?q=p??:Preprocessing2F
Iterator::Model9??v????!x1?&???)9??v????1x1?&???:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2?~j?t???!ʽ?3????)?~j?t???1ʽ?3????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchj?t???![o?ӵ???)j?t???1[o?ӵ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9h?㢩?b?I9??r??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	=
ףp=??=
ףp=??!=
ףp=??      ??!       "      ??!       *      ??!       2	+?????@+?????@!+?????@:      ??!       B      ??!       J	??(\?µ???(\?µ?!??(\?µ?R      ??!       Z	??(\?µ???(\?µ?!??(\?µ?b      ??!       JCPU_ONLYYh?㢩?b?b q9??r??X@Y      Y@qiyF??N?"?
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