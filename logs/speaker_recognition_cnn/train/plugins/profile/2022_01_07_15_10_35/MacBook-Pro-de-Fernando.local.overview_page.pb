?	???Qǫ@???Qǫ@!???Qǫ@	??ꂽa???ꂽa?!??ꂽa?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???Qǫ@??n????A9??v?ū@Y??ʡE???*	    ?BA2?
^Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Zip[0]::Map?-??吏h@!???˄?X@)??Q??h@1??????X@:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Zip??rh???h@!????X@)Zd;?O???1??1%????:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Zip[1]::TensorSlicePq=
ףp??! ???N???)q=
ףp??1 ???N???:Preprocessing2?
kIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::Shuffle::Zip[0]::Map::TensorSlice??O??n??!2Z??C>??)?O??n??12Z??C>??:Preprocessing2?
QIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::BatchV2::ShuffleP???K7?b@!?r??R@)P??n???1??9?Ν??:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2??|?5^??!?L.?截?)??|?5^??1?L.?截?:Preprocessing2F
Iterator::Modely?&1???!?I<?tڜ?)y?&1???1?I<?tڜ?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???Mb??!,P ?X7??)???Mb??1,P ?X7??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??ꂽa?I?*????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??n??????n????!??n????      ??!       "      ??!       *      ??!       2	9??v?ū@9??v?ū@!9??v?ū@:      ??!       B      ??!       J	??ʡE?????ʡE???!??ʡE???R      ??!       Z	??ʡE?????ʡE???!??ʡE???b      ??!       JCPU_ONLYY??ꂽa?b q?*????X@Y      Y@q????f`?"?
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