?	? ?r?ì@? ?r?ì@!? ?r?ì@	?"????u??"????u?!?"????u?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$? ?r?ì@X9??v???A??K7	¬@Y?E??????*	    ??@2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip[0]::Map?Zd;?Y@!??/!nX@)??Q?Y@1^UI?TX@:Preprocessing2?
`Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::ShuffleJ?K7?AO@!$?ږ4N@)+?????1??R
??:Preprocessing2?
eIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip?l???AY@!G?????X@)???K7???1D
??????:Preprocessing2?
zIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip[0]::Map::TensorSlice???n????!ِ??A??)??n????1ِ??A??:Preprocessing2?
uIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2::BatchV2::Shuffle::Zip[1]::TensorSliceJV-????!??m????)V-????1??m????:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2::ParallelMapV2bX9?ȶ?!xE???'??)bX9?ȶ?1xE???'??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??ʡE???!ɵ??*??)??ʡE???1ɵ??*??:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Prefetch::ParallelMapV2?l??????!??=6?k??)?l??????1??=6?k??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismL7?A`???!S?R??m??))\???(??16?;?a??:Preprocessing2F
Iterator::Model???(\???!d????)9??v????1????d???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?"????u?I?5u???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	X9??v???X9??v???!X9??v???      ??!       "      ??!       *      ??!       2	??K7	¬@??K7	¬@!??K7	¬@:      ??!       B      ??!       J	?E???????E??????!?E??????R      ??!       Z	?E???????E??????!?E??????b      ??!       JCPU_ONLYY?"????u?b q?5u???X@Y      Y@q?_??W?"?
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