��

��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02unknown8ܜ	
u
preds/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *
shared_namepreds/kernel
n
 preds/kernel/Read/ReadVariableOpReadVariableOppreds/kernel*
_output_shapes
:	� *
dtype0
l

preds/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
preds/bias
e
preds/bias/Read/ReadVariableOpReadVariableOp
preds/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@ *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�I� *
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
�I� *
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:� *
dtype0

NoOpNoOp
�(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�'
value�'B�' B�'
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	variables
trainable_variables
regularization_losses
		keras_api


signatures
 
 
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
layer_with_weights-3
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
F
"0
#1
$2
%3
&4
'5
(6
)7
8
9
F
"0
#1
$2
%3
&4
'5
(6
)7
8
9
 
�
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
 
 
h

"kernel
#bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
R
3	variables
4trainable_variables
5regularization_losses
6	keras_api
h

$kernel
%bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
R
;	variables
<trainable_variables
=regularization_losses
>	keras_api
h

&kernel
'bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
R
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
R
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
h

(kernel
)bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
8
"0
#1
$2
%3
&4
'5
(6
)7
8
"0
#1
$2
%3
&4
'5
(6
)7
 
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
XV
VARIABLE_VALUEpreds/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
preds/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
 regularization_losses
IG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4
 
 
 

"0
#1

"0
#1
 
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
/	variables
0trainable_variables
1regularization_losses
 
 
 
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
3	variables
4trainable_variables
5regularization_losses

$0
%1

$0
%1
 
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
7	variables
8trainable_variables
9regularization_losses
 
 
 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
;	variables
<trainable_variables
=regularization_losses

&0
'1

&0
'1
 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
 
 
 
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
 
 
 
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses

(0
)1

(0
)1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
 
?
0
1
2
3
4
5
6
7
8
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_input_2Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
serving_default_input_3Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2serving_default_input_3conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biaspreds/kernel
preds/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference_signature_wrapper_992
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename preds/kernel/Read/ReadVariableOppreds/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *&
f!R
__inference__traced_save_1555
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamepreds/kernel
preds/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_restore_1595��
�
�
__inference__traced_save_1555
file_prefix+
'savev2_preds_kernel_read_readvariableop)
%savev2_preds_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_preds_kernel_read_readvariableop%savev2_preds_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapesr
p: :	� ::@:@:@ : : ::
�I� :� : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	� : 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::&	"
 
_output_shapes
:
�I� :!


_output_shapes	
:� :

_output_shapes
: 
�

�
?__inference_dense_layer_call_and_return_conditional_losses_1501

inputs2
matmul_readvariableop_resource:
�I� .
biasadd_readvariableop_resource:	� 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�I� *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:���������� b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:���������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������I: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������I
 
_user_specified_nameinputs
�*
�
 __inference__traced_restore_1595
file_prefix0
assignvariableop_preds_kernel:	� +
assignvariableop_1_preds_bias::
 assignvariableop_2_conv2d_kernel:@,
assignvariableop_3_conv2d_bias:@<
"assignvariableop_4_conv2d_1_kernel:@ .
 assignvariableop_5_conv2d_1_bias: <
"assignvariableop_6_conv2d_2_kernel: .
 assignvariableop_7_conv2d_2_bias:3
assignvariableop_8_dense_kernel:
�I� ,
assignvariableop_9_dense_bias:	� 
identity_11��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_preds_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_preds_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
J
.__inference_max_pooling2d_2_layer_call_fn_1455

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_328�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
\
@__inference_flatten_layer_call_and_return_conditional_losses_413

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����$  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������IY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������I"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1385

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_1044
inputs_0
inputs_1!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: 
	unknown_4:
	unknown_5:
�I� 
	unknown_6:	� 
	unknown_7:	� 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_841o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:�����������:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
�
A__inference_conv2d_2_layer_call_and_return_conditional_losses_395

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������88i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������88 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������88 
 
_user_specified_nameinputs
�
�
>__inference_model_layer_call_and_return_conditional_losses_964
input_2
input_3'
embedding_931:@
embedding_933:@'
embedding_935:@ 
embedding_937: '
embedding_939: 
embedding_941:!
embedding_943:
�I� 
embedding_945:	� 
	preds_958:	� 
	preds_960:
identity��!embedding/StatefulPartitionedCall�#embedding/StatefulPartitionedCall_1�preds/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_931embedding_933embedding_935embedding_937embedding_939embedding_941embedding_943embedding_945*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_564�
#embedding/StatefulPartitionedCall_1StatefulPartitionedCallinput_3embedding_931embedding_933embedding_935embedding_937embedding_939embedding_941embedding_943embedding_945*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_564�
lambda/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_772�
preds/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0	preds_958	preds_960*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_preds_layer_call_and_return_conditional_losses_715u
IdentityIdentity&preds/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^embedding/StatefulPartitionedCall$^embedding/StatefulPartitionedCall_1^preds/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:�����������:�����������: : : : : : : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding/StatefulPartitionedCall_1#embedding/StatefulPartitionedCall_12>
preds/StatefulPartitionedCallpreds/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2:ZV
1
_output_shapes
:�����������
!
_user_specified_name	input_3
�
J
.__inference_max_pooling2d_1_layer_call_fn_1420

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_382h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������88 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������pp :W S
/
_output_shapes
:���������pp 
 
_user_specified_nameinputs
�
J
.__inference_max_pooling2d_2_layer_call_fn_1460

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_405h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������88:W S
/
_output_shapes
:���������88
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_1018
inputs_0
inputs_1!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: 
	unknown_4:
	unknown_5:
�I� 
	unknown_6:	� 
	unknown_7:	� 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_722o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:�����������:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_1481

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����$  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������IY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������I"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�j
�
?__inference_model_layer_call_and_return_conditional_losses_1116
inputs_0
inputs_1I
/embedding_conv2d_conv2d_readvariableop_resource:@>
0embedding_conv2d_biasadd_readvariableop_resource:@K
1embedding_conv2d_1_conv2d_readvariableop_resource:@ @
2embedding_conv2d_1_biasadd_readvariableop_resource: K
1embedding_conv2d_2_conv2d_readvariableop_resource: @
2embedding_conv2d_2_biasadd_readvariableop_resource:B
.embedding_dense_matmul_readvariableop_resource:
�I� >
/embedding_dense_biasadd_readvariableop_resource:	� 7
$preds_matmul_readvariableop_resource:	� 3
%preds_biasadd_readvariableop_resource:
identity��'embedding/conv2d/BiasAdd/ReadVariableOp�)embedding/conv2d/BiasAdd_1/ReadVariableOp�&embedding/conv2d/Conv2D/ReadVariableOp�(embedding/conv2d/Conv2D_1/ReadVariableOp�)embedding/conv2d_1/BiasAdd/ReadVariableOp�+embedding/conv2d_1/BiasAdd_1/ReadVariableOp�(embedding/conv2d_1/Conv2D/ReadVariableOp�*embedding/conv2d_1/Conv2D_1/ReadVariableOp�)embedding/conv2d_2/BiasAdd/ReadVariableOp�+embedding/conv2d_2/BiasAdd_1/ReadVariableOp�(embedding/conv2d_2/Conv2D/ReadVariableOp�*embedding/conv2d_2/Conv2D_1/ReadVariableOp�&embedding/dense/BiasAdd/ReadVariableOp�(embedding/dense/BiasAdd_1/ReadVariableOp�%embedding/dense/MatMul/ReadVariableOp�'embedding/dense/MatMul_1/ReadVariableOp�preds/BiasAdd/ReadVariableOp�preds/MatMul/ReadVariableOp�
&embedding/conv2d/Conv2D/ReadVariableOpReadVariableOp/embedding_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
embedding/conv2d/Conv2DConv2Dinputs_0.embedding/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
'embedding/conv2d/BiasAdd/ReadVariableOpReadVariableOp0embedding_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
embedding/conv2d/BiasAddBiasAdd embedding/conv2d/Conv2D:output:0/embedding/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@|
embedding/conv2d/ReluRelu!embedding/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
embedding/max_pooling2d/MaxPoolMaxPool#embedding/conv2d/Relu:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
�
(embedding/conv2d_1/Conv2D/ReadVariableOpReadVariableOp1embedding_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
embedding/conv2d_1/Conv2DConv2D(embedding/max_pooling2d/MaxPool:output:00embedding/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp *
paddingSAME*
strides
�
)embedding/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2embedding_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
embedding/conv2d_1/BiasAddBiasAdd"embedding/conv2d_1/Conv2D:output:01embedding/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp ~
embedding/conv2d_1/ReluRelu#embedding/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp �
!embedding/max_pooling2d_1/MaxPoolMaxPool%embedding/conv2d_1/Relu:activations:0*/
_output_shapes
:���������88 *
ksize
*
paddingVALID*
strides
�
(embedding/conv2d_2/Conv2D/ReadVariableOpReadVariableOp1embedding_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
embedding/conv2d_2/Conv2DConv2D*embedding/max_pooling2d_1/MaxPool:output:00embedding/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
)embedding/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2embedding_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
embedding/conv2d_2/BiasAddBiasAdd"embedding/conv2d_2/Conv2D:output:01embedding/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88~
embedding/conv2d_2/ReluRelu#embedding/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
!embedding/max_pooling2d_2/MaxPoolMaxPool%embedding/conv2d_2/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
h
embedding/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����$  �
embedding/flatten/ReshapeReshape*embedding/max_pooling2d_2/MaxPool:output:0 embedding/flatten/Const:output:0*
T0*(
_output_shapes
:����������I�
%embedding/dense/MatMul/ReadVariableOpReadVariableOp.embedding_dense_matmul_readvariableop_resource* 
_output_shapes
:
�I� *
dtype0�
embedding/dense/MatMulMatMul"embedding/flatten/Reshape:output:0-embedding/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
&embedding/dense/BiasAdd/ReadVariableOpReadVariableOp/embedding_dense_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
embedding/dense/BiasAddBiasAdd embedding/dense/MatMul:product:0.embedding/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� q
embedding/dense/ReluRelu embedding/dense/BiasAdd:output:0*
T0*(
_output_shapes
:���������� �
(embedding/conv2d/Conv2D_1/ReadVariableOpReadVariableOp/embedding_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
embedding/conv2d/Conv2D_1Conv2Dinputs_10embedding/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
)embedding/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp0embedding_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
embedding/conv2d/BiasAdd_1BiasAdd"embedding/conv2d/Conv2D_1:output:01embedding/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
embedding/conv2d/Relu_1Relu#embedding/conv2d/BiasAdd_1:output:0*
T0*1
_output_shapes
:�����������@�
!embedding/max_pooling2d/MaxPool_1MaxPool%embedding/conv2d/Relu_1:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
�
*embedding/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp1embedding_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
embedding/conv2d_1/Conv2D_1Conv2D*embedding/max_pooling2d/MaxPool_1:output:02embedding/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp *
paddingSAME*
strides
�
+embedding/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp2embedding_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
embedding/conv2d_1/BiasAdd_1BiasAdd$embedding/conv2d_1/Conv2D_1:output:03embedding/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp �
embedding/conv2d_1/Relu_1Relu%embedding/conv2d_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������pp �
#embedding/max_pooling2d_1/MaxPool_1MaxPool'embedding/conv2d_1/Relu_1:activations:0*/
_output_shapes
:���������88 *
ksize
*
paddingVALID*
strides
�
*embedding/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp1embedding_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
embedding/conv2d_2/Conv2D_1Conv2D,embedding/max_pooling2d_1/MaxPool_1:output:02embedding/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
+embedding/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp2embedding_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
embedding/conv2d_2/BiasAdd_1BiasAdd$embedding/conv2d_2/Conv2D_1:output:03embedding/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88�
embedding/conv2d_2/Relu_1Relu%embedding/conv2d_2/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������88�
#embedding/max_pooling2d_2/MaxPool_1MaxPool'embedding/conv2d_2/Relu_1:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
j
embedding/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"�����$  �
embedding/flatten/Reshape_1Reshape,embedding/max_pooling2d_2/MaxPool_1:output:0"embedding/flatten/Const_1:output:0*
T0*(
_output_shapes
:����������I�
'embedding/dense/MatMul_1/ReadVariableOpReadVariableOp.embedding_dense_matmul_readvariableop_resource* 
_output_shapes
:
�I� *
dtype0�
embedding/dense/MatMul_1MatMul$embedding/flatten/Reshape_1:output:0/embedding/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
(embedding/dense/BiasAdd_1/ReadVariableOpReadVariableOp/embedding_dense_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
embedding/dense/BiasAdd_1BiasAdd"embedding/dense/MatMul_1:product:00embedding/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� u
embedding/dense/Relu_1Relu"embedding/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:���������� �

lambda/subSub"embedding/dense/Relu:activations:0$embedding/dense/Relu_1:activations:0*
T0*(
_output_shapes
:���������� Z
lambda/SquareSquarelambda/sub:z:0*
T0*(
_output_shapes
:���������� �
preds/MatMul/ReadVariableOpReadVariableOp$preds_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
preds/MatMulMatMullambda/Square:y:0#preds/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
preds/BiasAdd/ReadVariableOpReadVariableOp%preds_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
preds/BiasAddBiasAddpreds/MatMul:product:0$preds/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
preds/SigmoidSigmoidpreds/BiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitypreds/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^embedding/conv2d/BiasAdd/ReadVariableOp*^embedding/conv2d/BiasAdd_1/ReadVariableOp'^embedding/conv2d/Conv2D/ReadVariableOp)^embedding/conv2d/Conv2D_1/ReadVariableOp*^embedding/conv2d_1/BiasAdd/ReadVariableOp,^embedding/conv2d_1/BiasAdd_1/ReadVariableOp)^embedding/conv2d_1/Conv2D/ReadVariableOp+^embedding/conv2d_1/Conv2D_1/ReadVariableOp*^embedding/conv2d_2/BiasAdd/ReadVariableOp,^embedding/conv2d_2/BiasAdd_1/ReadVariableOp)^embedding/conv2d_2/Conv2D/ReadVariableOp+^embedding/conv2d_2/Conv2D_1/ReadVariableOp'^embedding/dense/BiasAdd/ReadVariableOp)^embedding/dense/BiasAdd_1/ReadVariableOp&^embedding/dense/MatMul/ReadVariableOp(^embedding/dense/MatMul_1/ReadVariableOp^preds/BiasAdd/ReadVariableOp^preds/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:�����������:�����������: : : : : : : : : : 2R
'embedding/conv2d/BiasAdd/ReadVariableOp'embedding/conv2d/BiasAdd/ReadVariableOp2V
)embedding/conv2d/BiasAdd_1/ReadVariableOp)embedding/conv2d/BiasAdd_1/ReadVariableOp2P
&embedding/conv2d/Conv2D/ReadVariableOp&embedding/conv2d/Conv2D/ReadVariableOp2T
(embedding/conv2d/Conv2D_1/ReadVariableOp(embedding/conv2d/Conv2D_1/ReadVariableOp2V
)embedding/conv2d_1/BiasAdd/ReadVariableOp)embedding/conv2d_1/BiasAdd/ReadVariableOp2Z
+embedding/conv2d_1/BiasAdd_1/ReadVariableOp+embedding/conv2d_1/BiasAdd_1/ReadVariableOp2T
(embedding/conv2d_1/Conv2D/ReadVariableOp(embedding/conv2d_1/Conv2D/ReadVariableOp2X
*embedding/conv2d_1/Conv2D_1/ReadVariableOp*embedding/conv2d_1/Conv2D_1/ReadVariableOp2V
)embedding/conv2d_2/BiasAdd/ReadVariableOp)embedding/conv2d_2/BiasAdd/ReadVariableOp2Z
+embedding/conv2d_2/BiasAdd_1/ReadVariableOp+embedding/conv2d_2/BiasAdd_1/ReadVariableOp2T
(embedding/conv2d_2/Conv2D/ReadVariableOp(embedding/conv2d_2/Conv2D/ReadVariableOp2X
*embedding/conv2d_2/Conv2D_1/ReadVariableOp*embedding/conv2d_2/Conv2D_1/ReadVariableOp2P
&embedding/dense/BiasAdd/ReadVariableOp&embedding/dense/BiasAdd/ReadVariableOp2T
(embedding/dense/BiasAdd_1/ReadVariableOp(embedding/dense/BiasAdd_1/ReadVariableOp2N
%embedding/dense/MatMul/ReadVariableOp%embedding/dense/MatMul/ReadVariableOp2R
'embedding/dense/MatMul_1/ReadVariableOp'embedding/dense/MatMul_1/ReadVariableOp2<
preds/BiasAdd/ReadVariableOppreds/BiasAdd/ReadVariableOp2:
preds/MatMul/ReadVariableOppreds/MatMul/ReadVariableOp:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
�
$__inference_preds_layer_call_fn_1339

inputs
unknown:	� 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_preds_layer_call_and_return_conditional_losses_715o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
B
&__inference_flatten_layer_call_fn_1475

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������I* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_413a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������I"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_model_layer_call_fn_745
input_2
input_3!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: 
	unknown_4:
	unknown_5:
�I� 
	unknown_6:	� 
	unknown_7:	� 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_722o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:�����������:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2:ZV
1
_output_shapes
:�����������
!
_user_specified_name	input_3
�!
�
B__inference_embedding_layer_call_and_return_conditional_losses_632
input_1$

conv2d_607:@

conv2d_609:@&
conv2d_1_613:@ 
conv2d_1_615: &
conv2d_2_619: 
conv2d_2_621:
	dense_626:
�I� 
	dense_628:	� 
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1
conv2d_607
conv2d_609*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_349�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_359�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_613conv2d_1_615*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_372�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_382�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_619conv2d_2_621*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_395�
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_405�
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������I* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_413�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	dense_626	dense_628*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_426v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� �
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:�����������: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
#__inference_model_layer_call_fn_890
input_2
input_3!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: 
	unknown_4:
	unknown_5:
�I� 
	unknown_6:	� 
	unknown_7:	� 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_841o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:�����������:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2:ZV
1
_output_shapes
:�����������
!
_user_specified_name	input_3
�
�
A__inference_conv2d_1_layer_call_and_return_conditional_losses_372

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������pp i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������pp w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������pp@
 
_user_specified_nameinputs
�
d
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_405

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������88:W S
/
_output_shapes
:���������88
 
_user_specified_nameinputs
�	
�
'__inference_embedding_layer_call_fn_452
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: 
	unknown_4:
	unknown_5:
�I� 
	unknown_6:	� 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_433p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:�����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�	
�
'__inference_embedding_layer_call_fn_604
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: 
	unknown_4:
	unknown_5:
�I� 
	unknown_6:	� 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_564p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:�����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�

�
?__inference_preds_layer_call_and_return_conditional_losses_1350

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
>__inference_model_layer_call_and_return_conditional_losses_722

inputs
inputs_1'
embedding_669:@
embedding_671:@'
embedding_673:@ 
embedding_675: '
embedding_677: 
embedding_679:!
embedding_681:
�I� 
embedding_683:	� 
	preds_716:	� 
	preds_718:
identity��!embedding/StatefulPartitionedCall�#embedding/StatefulPartitionedCall_1�preds/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_669embedding_671embedding_673embedding_675embedding_677embedding_679embedding_681embedding_683*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_433�
#embedding/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1embedding_669embedding_671embedding_673embedding_675embedding_677embedding_679embedding_681embedding_683*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_433�
lambda/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_702�
preds/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0	preds_716	preds_718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_preds_layer_call_and_return_conditional_losses_715u
IdentityIdentity&preds/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^embedding/StatefulPartitionedCall$^embedding/StatefulPartitionedCall_1^preds/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:�����������:�����������: : : : : : : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding/StatefulPartitionedCall_1#embedding/StatefulPartitionedCall_12>
preds/StatefulPartitionedCallpreds/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:YU
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
d
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_382

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������88 *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������88 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������pp :W S
/
_output_shapes
:���������pp 
 
_user_specified_nameinputs
�
H
,__inference_max_pooling2d_layer_call_fn_1380

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_359h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������pp@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������@:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�!
�
B__inference_embedding_layer_call_and_return_conditional_losses_660
input_1$

conv2d_635:@

conv2d_637:@&
conv2d_1_641:@ 
conv2d_1_643: &
conv2d_2_647: 
conv2d_2_649:
	dense_654:
�I� 
	dense_656:	� 
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1
conv2d_635
conv2d_637*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_349�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_359�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_641conv2d_1_643*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_372�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_382�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_647conv2d_2_649*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_395�
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_405�
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������I* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_413�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	dense_654	dense_656*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_426v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� �
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:�����������: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
'__inference_conv2d_2_layer_call_fn_1439

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_395w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������88 : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������88 
 
_user_specified_nameinputs
�
�
$__inference_dense_layer_call_fn_1490

inputs
unknown:
�I� 
	unknown_0:	� 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_426p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������I: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������I
 
_user_specified_nameinputs
�
�
>__inference_model_layer_call_and_return_conditional_losses_927
input_2
input_3'
embedding_894:@
embedding_896:@'
embedding_898:@ 
embedding_900: '
embedding_902: 
embedding_904:!
embedding_906:
�I� 
embedding_908:	� 
	preds_921:	� 
	preds_923:
identity��!embedding/StatefulPartitionedCall�#embedding/StatefulPartitionedCall_1�preds/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_894embedding_896embedding_898embedding_900embedding_902embedding_904embedding_906embedding_908*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_433�
#embedding/StatefulPartitionedCall_1StatefulPartitionedCallinput_3embedding_894embedding_896embedding_898embedding_900embedding_902embedding_904embedding_906embedding_908*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_433�
lambda/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_702�
preds/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0	preds_921	preds_923*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_preds_layer_call_and_return_conditional_losses_715u
IdentityIdentity&preds/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^embedding/StatefulPartitionedCall$^embedding/StatefulPartitionedCall_1^preds/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:�����������:�����������: : : : : : : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding/StatefulPartitionedCall_1#embedding/StatefulPartitionedCall_12>
preds/StatefulPartitionedCallpreds/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2:ZV
1
_output_shapes
:�����������
!
_user_specified_name	input_3
�
�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1410

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������pp i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������pp w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������pp@
 
_user_specified_nameinputs
�
�
%__inference_conv2d_layer_call_fn_1359

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_349y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
'__inference_conv2d_1_layer_call_fn_1399

inputs!
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_372w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������pp `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������pp@
 
_user_specified_nameinputs
�!
�
B__inference_embedding_layer_call_and_return_conditional_losses_433

inputs$

conv2d_350:@

conv2d_352:@&
conv2d_1_373:@ 
conv2d_1_375: &
conv2d_2_396: 
conv2d_2_398:
	dense_427:
�I� 
	dense_429:	� 
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs
conv2d_350
conv2d_352*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_349�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_359�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_373conv2d_1_375*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_372�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_382�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_396conv2d_2_398*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_395�
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_405�
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������I* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_413�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	dense_427	dense_429*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_426v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� �
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:�����������: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
b
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_304

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
(__inference_embedding_layer_call_fn_1209

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: 
	unknown_4:
	unknown_5:
�I� 
	unknown_6:	� 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_433p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:�����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
Q
%__inference_lambda_layer_call_fn_1316
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_772a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:���������� :���������� :R N
(
_output_shapes
:���������� 
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:���������� 
"
_user_specified_name
inputs/1
�
�
@__inference_conv2d_layer_call_and_return_conditional_losses_1370

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1430

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������88 *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������88 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������pp :W S
/
_output_shapes
:���������pp 
 
_user_specified_nameinputs
�
d
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_328

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1465

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
b
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_359

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������pp@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������@:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1450

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������88i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������88 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������88 
 
_user_specified_nameinputs
�

�
>__inference_preds_layer_call_and_return_conditional_losses_715

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
H
,__inference_max_pooling2d_layer_call_fn_1375

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_304�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�+
�
C__inference_embedding_layer_call_and_return_conditional_losses_1304

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@A
'conv2d_1_conv2d_readvariableop_resource:@ 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource:8
$dense_matmul_readvariableop_resource:
�I� 4
%dense_biasadd_readvariableop_resource:	� 
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@h
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp *
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp �
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:���������88 *
ksize
*
paddingVALID*
strides
�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����$  �
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������I�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
�I� *
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� 
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:���������� h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:���������� �
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:�����������: : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
>__inference_model_layer_call_and_return_conditional_losses_841

inputs
inputs_1'
embedding_808:@
embedding_810:@'
embedding_812:@ 
embedding_814: '
embedding_816: 
embedding_818:!
embedding_820:
�I� 
embedding_822:	� 
	preds_835:	� 
	preds_837:
identity��!embedding/StatefulPartitionedCall�#embedding/StatefulPartitionedCall_1�preds/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_808embedding_810embedding_812embedding_814embedding_816embedding_818embedding_820embedding_822*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_564�
#embedding/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1embedding_808embedding_810embedding_812embedding_814embedding_816embedding_818embedding_820embedding_822*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_564�
lambda/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_772�
preds/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0	preds_835	preds_837*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_preds_layer_call_and_return_conditional_losses_715u
IdentityIdentity&preds/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^embedding/StatefulPartitionedCall$^embedding/StatefulPartitionedCall_1^preds/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:�����������:�����������: : : : : : : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding/StatefulPartitionedCall_1#embedding/StatefulPartitionedCall_12>
preds/StatefulPartitionedCallpreds/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:YU
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
i
?__inference_lambda_layer_call_and_return_conditional_losses_772

inputs
inputs_1
identityO
subSubinputsinputs_1*
T0*(
_output_shapes
:���������� L
SquareSquaresub:z:0*
T0*(
_output_shapes
:���������� S
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:���������� :���������� :P L
(
_output_shapes
:���������� 
 
_user_specified_nameinputs:PL
(
_output_shapes
:���������� 
 
_user_specified_nameinputs
�j
�
?__inference_model_layer_call_and_return_conditional_losses_1188
inputs_0
inputs_1I
/embedding_conv2d_conv2d_readvariableop_resource:@>
0embedding_conv2d_biasadd_readvariableop_resource:@K
1embedding_conv2d_1_conv2d_readvariableop_resource:@ @
2embedding_conv2d_1_biasadd_readvariableop_resource: K
1embedding_conv2d_2_conv2d_readvariableop_resource: @
2embedding_conv2d_2_biasadd_readvariableop_resource:B
.embedding_dense_matmul_readvariableop_resource:
�I� >
/embedding_dense_biasadd_readvariableop_resource:	� 7
$preds_matmul_readvariableop_resource:	� 3
%preds_biasadd_readvariableop_resource:
identity��'embedding/conv2d/BiasAdd/ReadVariableOp�)embedding/conv2d/BiasAdd_1/ReadVariableOp�&embedding/conv2d/Conv2D/ReadVariableOp�(embedding/conv2d/Conv2D_1/ReadVariableOp�)embedding/conv2d_1/BiasAdd/ReadVariableOp�+embedding/conv2d_1/BiasAdd_1/ReadVariableOp�(embedding/conv2d_1/Conv2D/ReadVariableOp�*embedding/conv2d_1/Conv2D_1/ReadVariableOp�)embedding/conv2d_2/BiasAdd/ReadVariableOp�+embedding/conv2d_2/BiasAdd_1/ReadVariableOp�(embedding/conv2d_2/Conv2D/ReadVariableOp�*embedding/conv2d_2/Conv2D_1/ReadVariableOp�&embedding/dense/BiasAdd/ReadVariableOp�(embedding/dense/BiasAdd_1/ReadVariableOp�%embedding/dense/MatMul/ReadVariableOp�'embedding/dense/MatMul_1/ReadVariableOp�preds/BiasAdd/ReadVariableOp�preds/MatMul/ReadVariableOp�
&embedding/conv2d/Conv2D/ReadVariableOpReadVariableOp/embedding_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
embedding/conv2d/Conv2DConv2Dinputs_0.embedding/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
'embedding/conv2d/BiasAdd/ReadVariableOpReadVariableOp0embedding_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
embedding/conv2d/BiasAddBiasAdd embedding/conv2d/Conv2D:output:0/embedding/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@|
embedding/conv2d/ReluRelu!embedding/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
embedding/max_pooling2d/MaxPoolMaxPool#embedding/conv2d/Relu:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
�
(embedding/conv2d_1/Conv2D/ReadVariableOpReadVariableOp1embedding_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
embedding/conv2d_1/Conv2DConv2D(embedding/max_pooling2d/MaxPool:output:00embedding/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp *
paddingSAME*
strides
�
)embedding/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2embedding_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
embedding/conv2d_1/BiasAddBiasAdd"embedding/conv2d_1/Conv2D:output:01embedding/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp ~
embedding/conv2d_1/ReluRelu#embedding/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp �
!embedding/max_pooling2d_1/MaxPoolMaxPool%embedding/conv2d_1/Relu:activations:0*/
_output_shapes
:���������88 *
ksize
*
paddingVALID*
strides
�
(embedding/conv2d_2/Conv2D/ReadVariableOpReadVariableOp1embedding_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
embedding/conv2d_2/Conv2DConv2D*embedding/max_pooling2d_1/MaxPool:output:00embedding/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
)embedding/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2embedding_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
embedding/conv2d_2/BiasAddBiasAdd"embedding/conv2d_2/Conv2D:output:01embedding/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88~
embedding/conv2d_2/ReluRelu#embedding/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
!embedding/max_pooling2d_2/MaxPoolMaxPool%embedding/conv2d_2/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
h
embedding/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����$  �
embedding/flatten/ReshapeReshape*embedding/max_pooling2d_2/MaxPool:output:0 embedding/flatten/Const:output:0*
T0*(
_output_shapes
:����������I�
%embedding/dense/MatMul/ReadVariableOpReadVariableOp.embedding_dense_matmul_readvariableop_resource* 
_output_shapes
:
�I� *
dtype0�
embedding/dense/MatMulMatMul"embedding/flatten/Reshape:output:0-embedding/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
&embedding/dense/BiasAdd/ReadVariableOpReadVariableOp/embedding_dense_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
embedding/dense/BiasAddBiasAdd embedding/dense/MatMul:product:0.embedding/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� q
embedding/dense/ReluRelu embedding/dense/BiasAdd:output:0*
T0*(
_output_shapes
:���������� �
(embedding/conv2d/Conv2D_1/ReadVariableOpReadVariableOp/embedding_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
embedding/conv2d/Conv2D_1Conv2Dinputs_10embedding/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
)embedding/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp0embedding_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
embedding/conv2d/BiasAdd_1BiasAdd"embedding/conv2d/Conv2D_1:output:01embedding/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
embedding/conv2d/Relu_1Relu#embedding/conv2d/BiasAdd_1:output:0*
T0*1
_output_shapes
:�����������@�
!embedding/max_pooling2d/MaxPool_1MaxPool%embedding/conv2d/Relu_1:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
�
*embedding/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp1embedding_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
embedding/conv2d_1/Conv2D_1Conv2D*embedding/max_pooling2d/MaxPool_1:output:02embedding/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp *
paddingSAME*
strides
�
+embedding/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp2embedding_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
embedding/conv2d_1/BiasAdd_1BiasAdd$embedding/conv2d_1/Conv2D_1:output:03embedding/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp �
embedding/conv2d_1/Relu_1Relu%embedding/conv2d_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������pp �
#embedding/max_pooling2d_1/MaxPool_1MaxPool'embedding/conv2d_1/Relu_1:activations:0*/
_output_shapes
:���������88 *
ksize
*
paddingVALID*
strides
�
*embedding/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp1embedding_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
embedding/conv2d_2/Conv2D_1Conv2D,embedding/max_pooling2d_1/MaxPool_1:output:02embedding/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
+embedding/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp2embedding_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
embedding/conv2d_2/BiasAdd_1BiasAdd$embedding/conv2d_2/Conv2D_1:output:03embedding/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88�
embedding/conv2d_2/Relu_1Relu%embedding/conv2d_2/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������88�
#embedding/max_pooling2d_2/MaxPool_1MaxPool'embedding/conv2d_2/Relu_1:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
j
embedding/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"�����$  �
embedding/flatten/Reshape_1Reshape,embedding/max_pooling2d_2/MaxPool_1:output:0"embedding/flatten/Const_1:output:0*
T0*(
_output_shapes
:����������I�
'embedding/dense/MatMul_1/ReadVariableOpReadVariableOp.embedding_dense_matmul_readvariableop_resource* 
_output_shapes
:
�I� *
dtype0�
embedding/dense/MatMul_1MatMul$embedding/flatten/Reshape_1:output:0/embedding/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
(embedding/dense/BiasAdd_1/ReadVariableOpReadVariableOp/embedding_dense_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
embedding/dense/BiasAdd_1BiasAdd"embedding/dense/MatMul_1:product:00embedding/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� u
embedding/dense/Relu_1Relu"embedding/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:���������� �

lambda/subSub"embedding/dense/Relu:activations:0$embedding/dense/Relu_1:activations:0*
T0*(
_output_shapes
:���������� Z
lambda/SquareSquarelambda/sub:z:0*
T0*(
_output_shapes
:���������� �
preds/MatMul/ReadVariableOpReadVariableOp$preds_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
preds/MatMulMatMullambda/Square:y:0#preds/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
preds/BiasAdd/ReadVariableOpReadVariableOp%preds_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
preds/BiasAddBiasAddpreds/MatMul:product:0$preds/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
preds/SigmoidSigmoidpreds/BiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitypreds/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^embedding/conv2d/BiasAdd/ReadVariableOp*^embedding/conv2d/BiasAdd_1/ReadVariableOp'^embedding/conv2d/Conv2D/ReadVariableOp)^embedding/conv2d/Conv2D_1/ReadVariableOp*^embedding/conv2d_1/BiasAdd/ReadVariableOp,^embedding/conv2d_1/BiasAdd_1/ReadVariableOp)^embedding/conv2d_1/Conv2D/ReadVariableOp+^embedding/conv2d_1/Conv2D_1/ReadVariableOp*^embedding/conv2d_2/BiasAdd/ReadVariableOp,^embedding/conv2d_2/BiasAdd_1/ReadVariableOp)^embedding/conv2d_2/Conv2D/ReadVariableOp+^embedding/conv2d_2/Conv2D_1/ReadVariableOp'^embedding/dense/BiasAdd/ReadVariableOp)^embedding/dense/BiasAdd_1/ReadVariableOp&^embedding/dense/MatMul/ReadVariableOp(^embedding/dense/MatMul_1/ReadVariableOp^preds/BiasAdd/ReadVariableOp^preds/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:�����������:�����������: : : : : : : : : : 2R
'embedding/conv2d/BiasAdd/ReadVariableOp'embedding/conv2d/BiasAdd/ReadVariableOp2V
)embedding/conv2d/BiasAdd_1/ReadVariableOp)embedding/conv2d/BiasAdd_1/ReadVariableOp2P
&embedding/conv2d/Conv2D/ReadVariableOp&embedding/conv2d/Conv2D/ReadVariableOp2T
(embedding/conv2d/Conv2D_1/ReadVariableOp(embedding/conv2d/Conv2D_1/ReadVariableOp2V
)embedding/conv2d_1/BiasAdd/ReadVariableOp)embedding/conv2d_1/BiasAdd/ReadVariableOp2Z
+embedding/conv2d_1/BiasAdd_1/ReadVariableOp+embedding/conv2d_1/BiasAdd_1/ReadVariableOp2T
(embedding/conv2d_1/Conv2D/ReadVariableOp(embedding/conv2d_1/Conv2D/ReadVariableOp2X
*embedding/conv2d_1/Conv2D_1/ReadVariableOp*embedding/conv2d_1/Conv2D_1/ReadVariableOp2V
)embedding/conv2d_2/BiasAdd/ReadVariableOp)embedding/conv2d_2/BiasAdd/ReadVariableOp2Z
+embedding/conv2d_2/BiasAdd_1/ReadVariableOp+embedding/conv2d_2/BiasAdd_1/ReadVariableOp2T
(embedding/conv2d_2/Conv2D/ReadVariableOp(embedding/conv2d_2/Conv2D/ReadVariableOp2X
*embedding/conv2d_2/Conv2D_1/ReadVariableOp*embedding/conv2d_2/Conv2D_1/ReadVariableOp2P
&embedding/dense/BiasAdd/ReadVariableOp&embedding/dense/BiasAdd/ReadVariableOp2T
(embedding/dense/BiasAdd_1/ReadVariableOp(embedding/dense/BiasAdd_1/ReadVariableOp2N
%embedding/dense/MatMul/ReadVariableOp%embedding/dense/MatMul/ReadVariableOp2R
'embedding/dense/MatMul_1/ReadVariableOp'embedding/dense/MatMul_1/ReadVariableOp2<
preds/BiasAdd/ReadVariableOppreds/BiasAdd/ReadVariableOp2:
preds/MatMul/ReadVariableOppreds/MatMul/ReadVariableOp:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�+
�
C__inference_embedding_layer_call_and_return_conditional_losses_1267

inputs?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@A
'conv2d_1_conv2d_readvariableop_resource:@ 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource:8
$dense_matmul_readvariableop_resource:
�I� 4
%dense_biasadd_readvariableop_resource:	� 
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@h
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp *
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp �
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:���������88 *
ksize
*
paddingVALID*
strides
�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����$  �
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������I�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
�I� *
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� 
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:���������� h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:���������� �
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:�����������: : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
l
@__inference_lambda_layer_call_and_return_conditional_losses_1323
inputs_0
inputs_1
identityQ
subSubinputs_0inputs_1*
T0*(
_output_shapes
:���������� L
SquareSquaresub:z:0*
T0*(
_output_shapes
:���������� S
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:���������� :���������� :R N
(
_output_shapes
:���������� 
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:���������� 
"
_user_specified_name
inputs/1
�
J
.__inference_max_pooling2d_1_layer_call_fn_1415

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_316�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�!
�
B__inference_embedding_layer_call_and_return_conditional_losses_564

inputs$

conv2d_539:@

conv2d_541:@&
conv2d_1_545:@ 
conv2d_1_547: &
conv2d_2_551: 
conv2d_2_553:
	dense_558:
�I� 
	dense_560:	� 
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs
conv2d_539
conv2d_541*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_349�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_359�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_545conv2d_1_547*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_372�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_382�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_551conv2d_2_553*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_395�
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_405�
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������I* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_413�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	dense_558	dense_560*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_426v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� �
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:�����������: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1390

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������pp@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������@:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
!__inference_signature_wrapper_992
input_2
input_3!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: 
	unknown_4:
	unknown_5:
�I� 
	unknown_6:	� 
	unknown_7:	� 
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__wrapped_model_295o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:�����������:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2:ZV
1
_output_shapes
:�����������
!
_user_specified_name	input_3
�
l
@__inference_lambda_layer_call_and_return_conditional_losses_1330
inputs_0
inputs_1
identityQ
subSubinputs_0inputs_1*
T0*(
_output_shapes
:���������� L
SquareSquaresub:z:0*
T0*(
_output_shapes
:���������� S
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:���������� :���������� :R N
(
_output_shapes
:���������� 
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:���������� 
"
_user_specified_name
inputs/1
�	
�
(__inference_embedding_layer_call_fn_1230

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: 
	unknown_4:
	unknown_5:
�I� 
	unknown_6:	� 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� **
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_embedding_layer_call_and_return_conditional_losses_564p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:�����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
d
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_316

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1470

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������88:W S
/
_output_shapes
:���������88
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1425

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
Q
%__inference_lambda_layer_call_fn_1310
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_702a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:���������� :���������� :R N
(
_output_shapes
:���������� 
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:���������� 
"
_user_specified_name
inputs/1
�

�
>__inference_dense_layer_call_and_return_conditional_losses_426

inputs2
matmul_readvariableop_resource:
�I� .
biasadd_readvariableop_resource:	� 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�I� *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:���������� b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:���������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������I: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������I
 
_user_specified_nameinputs
�
i
?__inference_lambda_layer_call_and_return_conditional_losses_702

inputs
inputs_1
identityO
subSubinputsinputs_1*
T0*(
_output_shapes
:���������� L
SquareSquaresub:z:0*
T0*(
_output_shapes
:���������� S
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:���������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:���������� :���������� :P L
(
_output_shapes
:���������� 
 
_user_specified_nameinputs:PL
(
_output_shapes
:���������� 
 
_user_specified_nameinputs
�s
�
__inference__wrapped_model_295
input_2
input_3O
5model_embedding_conv2d_conv2d_readvariableop_resource:@D
6model_embedding_conv2d_biasadd_readvariableop_resource:@Q
7model_embedding_conv2d_1_conv2d_readvariableop_resource:@ F
8model_embedding_conv2d_1_biasadd_readvariableop_resource: Q
7model_embedding_conv2d_2_conv2d_readvariableop_resource: F
8model_embedding_conv2d_2_biasadd_readvariableop_resource:H
4model_embedding_dense_matmul_readvariableop_resource:
�I� D
5model_embedding_dense_biasadd_readvariableop_resource:	� =
*model_preds_matmul_readvariableop_resource:	� 9
+model_preds_biasadd_readvariableop_resource:
identity��-model/embedding/conv2d/BiasAdd/ReadVariableOp�/model/embedding/conv2d/BiasAdd_1/ReadVariableOp�,model/embedding/conv2d/Conv2D/ReadVariableOp�.model/embedding/conv2d/Conv2D_1/ReadVariableOp�/model/embedding/conv2d_1/BiasAdd/ReadVariableOp�1model/embedding/conv2d_1/BiasAdd_1/ReadVariableOp�.model/embedding/conv2d_1/Conv2D/ReadVariableOp�0model/embedding/conv2d_1/Conv2D_1/ReadVariableOp�/model/embedding/conv2d_2/BiasAdd/ReadVariableOp�1model/embedding/conv2d_2/BiasAdd_1/ReadVariableOp�.model/embedding/conv2d_2/Conv2D/ReadVariableOp�0model/embedding/conv2d_2/Conv2D_1/ReadVariableOp�,model/embedding/dense/BiasAdd/ReadVariableOp�.model/embedding/dense/BiasAdd_1/ReadVariableOp�+model/embedding/dense/MatMul/ReadVariableOp�-model/embedding/dense/MatMul_1/ReadVariableOp�"model/preds/BiasAdd/ReadVariableOp�!model/preds/MatMul/ReadVariableOp�
,model/embedding/conv2d/Conv2D/ReadVariableOpReadVariableOp5model_embedding_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
model/embedding/conv2d/Conv2DConv2Dinput_24model/embedding/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
-model/embedding/conv2d/BiasAdd/ReadVariableOpReadVariableOp6model_embedding_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/embedding/conv2d/BiasAddBiasAdd&model/embedding/conv2d/Conv2D:output:05model/embedding/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
model/embedding/conv2d/ReluRelu'model/embedding/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
%model/embedding/max_pooling2d/MaxPoolMaxPool)model/embedding/conv2d/Relu:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
�
.model/embedding/conv2d_1/Conv2D/ReadVariableOpReadVariableOp7model_embedding_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
model/embedding/conv2d_1/Conv2DConv2D.model/embedding/max_pooling2d/MaxPool:output:06model/embedding/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp *
paddingSAME*
strides
�
/model/embedding/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp8model_embedding_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
 model/embedding/conv2d_1/BiasAddBiasAdd(model/embedding/conv2d_1/Conv2D:output:07model/embedding/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp �
model/embedding/conv2d_1/ReluRelu)model/embedding/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp �
'model/embedding/max_pooling2d_1/MaxPoolMaxPool+model/embedding/conv2d_1/Relu:activations:0*/
_output_shapes
:���������88 *
ksize
*
paddingVALID*
strides
�
.model/embedding/conv2d_2/Conv2D/ReadVariableOpReadVariableOp7model_embedding_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
model/embedding/conv2d_2/Conv2DConv2D0model/embedding/max_pooling2d_1/MaxPool:output:06model/embedding/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
/model/embedding/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp8model_embedding_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
 model/embedding/conv2d_2/BiasAddBiasAdd(model/embedding/conv2d_2/Conv2D:output:07model/embedding/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88�
model/embedding/conv2d_2/ReluRelu)model/embedding/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
'model/embedding/max_pooling2d_2/MaxPoolMaxPool+model/embedding/conv2d_2/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
n
model/embedding/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����$  �
model/embedding/flatten/ReshapeReshape0model/embedding/max_pooling2d_2/MaxPool:output:0&model/embedding/flatten/Const:output:0*
T0*(
_output_shapes
:����������I�
+model/embedding/dense/MatMul/ReadVariableOpReadVariableOp4model_embedding_dense_matmul_readvariableop_resource* 
_output_shapes
:
�I� *
dtype0�
model/embedding/dense/MatMulMatMul(model/embedding/flatten/Reshape:output:03model/embedding/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
,model/embedding/dense/BiasAdd/ReadVariableOpReadVariableOp5model_embedding_dense_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
model/embedding/dense/BiasAddBiasAdd&model/embedding/dense/MatMul:product:04model/embedding/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� }
model/embedding/dense/ReluRelu&model/embedding/dense/BiasAdd:output:0*
T0*(
_output_shapes
:���������� �
.model/embedding/conv2d/Conv2D_1/ReadVariableOpReadVariableOp5model_embedding_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
model/embedding/conv2d/Conv2D_1Conv2Dinput_36model/embedding/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
/model/embedding/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp6model_embedding_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 model/embedding/conv2d/BiasAdd_1BiasAdd(model/embedding/conv2d/Conv2D_1:output:07model/embedding/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
model/embedding/conv2d/Relu_1Relu)model/embedding/conv2d/BiasAdd_1:output:0*
T0*1
_output_shapes
:�����������@�
'model/embedding/max_pooling2d/MaxPool_1MaxPool+model/embedding/conv2d/Relu_1:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
�
0model/embedding/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp7model_embedding_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
!model/embedding/conv2d_1/Conv2D_1Conv2D0model/embedding/max_pooling2d/MaxPool_1:output:08model/embedding/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp *
paddingSAME*
strides
�
1model/embedding/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp8model_embedding_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"model/embedding/conv2d_1/BiasAdd_1BiasAdd*model/embedding/conv2d_1/Conv2D_1:output:09model/embedding/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp �
model/embedding/conv2d_1/Relu_1Relu+model/embedding/conv2d_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������pp �
)model/embedding/max_pooling2d_1/MaxPool_1MaxPool-model/embedding/conv2d_1/Relu_1:activations:0*/
_output_shapes
:���������88 *
ksize
*
paddingVALID*
strides
�
0model/embedding/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp7model_embedding_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
!model/embedding/conv2d_2/Conv2D_1Conv2D2model/embedding/max_pooling2d_1/MaxPool_1:output:08model/embedding/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
1model/embedding/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp8model_embedding_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"model/embedding/conv2d_2/BiasAdd_1BiasAdd*model/embedding/conv2d_2/Conv2D_1:output:09model/embedding/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88�
model/embedding/conv2d_2/Relu_1Relu+model/embedding/conv2d_2/BiasAdd_1:output:0*
T0*/
_output_shapes
:���������88�
)model/embedding/max_pooling2d_2/MaxPool_1MaxPool-model/embedding/conv2d_2/Relu_1:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
p
model/embedding/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"�����$  �
!model/embedding/flatten/Reshape_1Reshape2model/embedding/max_pooling2d_2/MaxPool_1:output:0(model/embedding/flatten/Const_1:output:0*
T0*(
_output_shapes
:����������I�
-model/embedding/dense/MatMul_1/ReadVariableOpReadVariableOp4model_embedding_dense_matmul_readvariableop_resource* 
_output_shapes
:
�I� *
dtype0�
model/embedding/dense/MatMul_1MatMul*model/embedding/flatten/Reshape_1:output:05model/embedding/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
.model/embedding/dense/BiasAdd_1/ReadVariableOpReadVariableOp5model_embedding_dense_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
model/embedding/dense/BiasAdd_1BiasAdd(model/embedding/dense/MatMul_1:product:06model/embedding/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
model/embedding/dense/Relu_1Relu(model/embedding/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:���������� �
model/lambda/subSub(model/embedding/dense/Relu:activations:0*model/embedding/dense/Relu_1:activations:0*
T0*(
_output_shapes
:���������� f
model/lambda/SquareSquaremodel/lambda/sub:z:0*
T0*(
_output_shapes
:���������� �
!model/preds/MatMul/ReadVariableOpReadVariableOp*model_preds_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
model/preds/MatMulMatMulmodel/lambda/Square:y:0)model/preds/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model/preds/BiasAdd/ReadVariableOpReadVariableOp+model_preds_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/preds/BiasAddBiasAddmodel/preds/MatMul:product:0*model/preds/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
model/preds/SigmoidSigmoidmodel/preds/BiasAdd:output:0*
T0*'
_output_shapes
:���������f
IdentityIdentitymodel/preds/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^model/embedding/conv2d/BiasAdd/ReadVariableOp0^model/embedding/conv2d/BiasAdd_1/ReadVariableOp-^model/embedding/conv2d/Conv2D/ReadVariableOp/^model/embedding/conv2d/Conv2D_1/ReadVariableOp0^model/embedding/conv2d_1/BiasAdd/ReadVariableOp2^model/embedding/conv2d_1/BiasAdd_1/ReadVariableOp/^model/embedding/conv2d_1/Conv2D/ReadVariableOp1^model/embedding/conv2d_1/Conv2D_1/ReadVariableOp0^model/embedding/conv2d_2/BiasAdd/ReadVariableOp2^model/embedding/conv2d_2/BiasAdd_1/ReadVariableOp/^model/embedding/conv2d_2/Conv2D/ReadVariableOp1^model/embedding/conv2d_2/Conv2D_1/ReadVariableOp-^model/embedding/dense/BiasAdd/ReadVariableOp/^model/embedding/dense/BiasAdd_1/ReadVariableOp,^model/embedding/dense/MatMul/ReadVariableOp.^model/embedding/dense/MatMul_1/ReadVariableOp#^model/preds/BiasAdd/ReadVariableOp"^model/preds/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:�����������:�����������: : : : : : : : : : 2^
-model/embedding/conv2d/BiasAdd/ReadVariableOp-model/embedding/conv2d/BiasAdd/ReadVariableOp2b
/model/embedding/conv2d/BiasAdd_1/ReadVariableOp/model/embedding/conv2d/BiasAdd_1/ReadVariableOp2\
,model/embedding/conv2d/Conv2D/ReadVariableOp,model/embedding/conv2d/Conv2D/ReadVariableOp2`
.model/embedding/conv2d/Conv2D_1/ReadVariableOp.model/embedding/conv2d/Conv2D_1/ReadVariableOp2b
/model/embedding/conv2d_1/BiasAdd/ReadVariableOp/model/embedding/conv2d_1/BiasAdd/ReadVariableOp2f
1model/embedding/conv2d_1/BiasAdd_1/ReadVariableOp1model/embedding/conv2d_1/BiasAdd_1/ReadVariableOp2`
.model/embedding/conv2d_1/Conv2D/ReadVariableOp.model/embedding/conv2d_1/Conv2D/ReadVariableOp2d
0model/embedding/conv2d_1/Conv2D_1/ReadVariableOp0model/embedding/conv2d_1/Conv2D_1/ReadVariableOp2b
/model/embedding/conv2d_2/BiasAdd/ReadVariableOp/model/embedding/conv2d_2/BiasAdd/ReadVariableOp2f
1model/embedding/conv2d_2/BiasAdd_1/ReadVariableOp1model/embedding/conv2d_2/BiasAdd_1/ReadVariableOp2`
.model/embedding/conv2d_2/Conv2D/ReadVariableOp.model/embedding/conv2d_2/Conv2D/ReadVariableOp2d
0model/embedding/conv2d_2/Conv2D_1/ReadVariableOp0model/embedding/conv2d_2/Conv2D_1/ReadVariableOp2\
,model/embedding/dense/BiasAdd/ReadVariableOp,model/embedding/dense/BiasAdd/ReadVariableOp2`
.model/embedding/dense/BiasAdd_1/ReadVariableOp.model/embedding/dense/BiasAdd_1/ReadVariableOp2Z
+model/embedding/dense/MatMul/ReadVariableOp+model/embedding/dense/MatMul/ReadVariableOp2^
-model/embedding/dense/MatMul_1/ReadVariableOp-model/embedding/dense/MatMul_1/ReadVariableOp2H
"model/preds/BiasAdd/ReadVariableOp"model/preds/BiasAdd/ReadVariableOp2F
!model/preds/MatMul/ReadVariableOp!model/preds/MatMul/ReadVariableOp:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2:ZV
1
_output_shapes
:�����������
!
_user_specified_name	input_3
�
�
?__inference_conv2d_layer_call_and_return_conditional_losses_349

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_2:
serving_default_input_2:0�����������
E
input_3:
serving_default_input_3:0�����������9
preds0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	variables
trainable_variables
regularization_losses
		keras_api


signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
layer_with_weights-3
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_network
�
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
f
"0
#1
$2
%3
&4
'5
(6
)7
8
9"
trackable_list_wrapper
f
"0
#1
$2
%3
&4
'5
(6
)7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
"
_tf_keras_input_layer
�

"kernel
#bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

$kernel
%bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

&kernel
'bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

(kernel
)bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
X
"0
#1
$2
%3
&4
'5
(6
)7"
trackable_list_wrapper
X
"0
#1
$2
%3
&4
'5
(6
)7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	� 2preds/kernel
:2
preds/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
 regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%@2conv2d/kernel
:@2conv2d/bias
):'@ 2conv2d_1/kernel
: 2conv2d_1/bias
):' 2conv2d_2/kernel
:2conv2d_2/bias
 :
�I� 2dense/kernel
:� 2
dense/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
/	variables
0trainable_variables
1regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
3	variables
4trainable_variables
5regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
7	variables
8trainable_variables
9regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
;	variables
<trainable_variables
=regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�2�
#__inference_model_layer_call_fn_745
$__inference_model_layer_call_fn_1018
$__inference_model_layer_call_fn_1044
#__inference_model_layer_call_fn_890�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_model_layer_call_and_return_conditional_losses_1116
?__inference_model_layer_call_and_return_conditional_losses_1188
>__inference_model_layer_call_and_return_conditional_losses_927
>__inference_model_layer_call_and_return_conditional_losses_964�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
__inference__wrapped_model_295input_2input_3"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_embedding_layer_call_fn_452
(__inference_embedding_layer_call_fn_1209
(__inference_embedding_layer_call_fn_1230
'__inference_embedding_layer_call_fn_604�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_embedding_layer_call_and_return_conditional_losses_1267
C__inference_embedding_layer_call_and_return_conditional_losses_1304
B__inference_embedding_layer_call_and_return_conditional_losses_632
B__inference_embedding_layer_call_and_return_conditional_losses_660�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_lambda_layer_call_fn_1310
%__inference_lambda_layer_call_fn_1316�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_lambda_layer_call_and_return_conditional_losses_1323
@__inference_lambda_layer_call_and_return_conditional_losses_1330�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
$__inference_preds_layer_call_fn_1339�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_preds_layer_call_and_return_conditional_losses_1350�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
!__inference_signature_wrapper_992input_2input_3"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_conv2d_layer_call_fn_1359�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_conv2d_layer_call_and_return_conditional_losses_1370�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_max_pooling2d_layer_call_fn_1375
,__inference_max_pooling2d_layer_call_fn_1380�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1385
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1390�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv2d_1_layer_call_fn_1399�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1410�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_max_pooling2d_1_layer_call_fn_1415
.__inference_max_pooling2d_1_layer_call_fn_1420�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1425
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1430�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv2d_2_layer_call_fn_1439�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1450�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_max_pooling2d_2_layer_call_fn_1455
.__inference_max_pooling2d_2_layer_call_fn_1460�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1465
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1470�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_flatten_layer_call_fn_1475�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_flatten_layer_call_and_return_conditional_losses_1481�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_dense_layer_call_fn_1490�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_dense_layer_call_and_return_conditional_losses_1501�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference__wrapped_model_295�
"#$%&'()l�i
b�_
]�Z
+�(
input_2�����������
+�(
input_3�����������
� "-�*
(
preds�
preds����������
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1410l$%7�4
-�*
(�%
inputs���������pp@
� "-�*
#� 
0���������pp 
� �
'__inference_conv2d_1_layer_call_fn_1399_$%7�4
-�*
(�%
inputs���������pp@
� " ����������pp �
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1450l&'7�4
-�*
(�%
inputs���������88 
� "-�*
#� 
0���������88
� �
'__inference_conv2d_2_layer_call_fn_1439_&'7�4
-�*
(�%
inputs���������88 
� " ����������88�
@__inference_conv2d_layer_call_and_return_conditional_losses_1370p"#9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������@
� �
%__inference_conv2d_layer_call_fn_1359c"#9�6
/�,
*�'
inputs�����������
� ""������������@�
?__inference_dense_layer_call_and_return_conditional_losses_1501^()0�-
&�#
!�
inputs����������I
� "&�#
�
0���������� 
� y
$__inference_dense_layer_call_fn_1490Q()0�-
&�#
!�
inputs����������I
� "����������� �
C__inference_embedding_layer_call_and_return_conditional_losses_1267u"#$%&'()A�>
7�4
*�'
inputs�����������
p 

 
� "&�#
�
0���������� 
� �
C__inference_embedding_layer_call_and_return_conditional_losses_1304u"#$%&'()A�>
7�4
*�'
inputs�����������
p

 
� "&�#
�
0���������� 
� �
B__inference_embedding_layer_call_and_return_conditional_losses_632v"#$%&'()B�?
8�5
+�(
input_1�����������
p 

 
� "&�#
�
0���������� 
� �
B__inference_embedding_layer_call_and_return_conditional_losses_660v"#$%&'()B�?
8�5
+�(
input_1�����������
p

 
� "&�#
�
0���������� 
� �
(__inference_embedding_layer_call_fn_1209h"#$%&'()A�>
7�4
*�'
inputs�����������
p 

 
� "����������� �
(__inference_embedding_layer_call_fn_1230h"#$%&'()A�>
7�4
*�'
inputs�����������
p

 
� "����������� �
'__inference_embedding_layer_call_fn_452i"#$%&'()B�?
8�5
+�(
input_1�����������
p 

 
� "����������� �
'__inference_embedding_layer_call_fn_604i"#$%&'()B�?
8�5
+�(
input_1�����������
p

 
� "����������� �
A__inference_flatten_layer_call_and_return_conditional_losses_1481a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������I
� ~
&__inference_flatten_layer_call_fn_1475T7�4
-�*
(�%
inputs���������
� "�����������I�
@__inference_lambda_layer_call_and_return_conditional_losses_1323�d�a
Z�W
M�J
#� 
inputs/0���������� 
#� 
inputs/1���������� 

 
p 
� "&�#
�
0���������� 
� �
@__inference_lambda_layer_call_and_return_conditional_losses_1330�d�a
Z�W
M�J
#� 
inputs/0���������� 
#� 
inputs/1���������� 

 
p
� "&�#
�
0���������� 
� �
%__inference_lambda_layer_call_fn_1310�d�a
Z�W
M�J
#� 
inputs/0���������� 
#� 
inputs/1���������� 

 
p 
� "����������� �
%__inference_lambda_layer_call_fn_1316�d�a
Z�W
M�J
#� 
inputs/0���������� 
#� 
inputs/1���������� 

 
p
� "����������� �
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1425�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1430h7�4
-�*
(�%
inputs���������pp 
� "-�*
#� 
0���������88 
� �
.__inference_max_pooling2d_1_layer_call_fn_1415�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
.__inference_max_pooling2d_1_layer_call_fn_1420[7�4
-�*
(�%
inputs���������pp 
� " ����������88 �
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1465�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1470h7�4
-�*
(�%
inputs���������88
� "-�*
#� 
0���������
� �
.__inference_max_pooling2d_2_layer_call_fn_1455�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
.__inference_max_pooling2d_2_layer_call_fn_1460[7�4
-�*
(�%
inputs���������88
� " �����������
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1385�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1390j9�6
/�,
*�'
inputs�����������@
� "-�*
#� 
0���������pp@
� �
,__inference_max_pooling2d_layer_call_fn_1375�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
,__inference_max_pooling2d_layer_call_fn_1380]9�6
/�,
*�'
inputs�����������@
� " ����������pp@�
?__inference_model_layer_call_and_return_conditional_losses_1116�
"#$%&'()v�s
l�i
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
p 

 
� "%�"
�
0���������
� �
?__inference_model_layer_call_and_return_conditional_losses_1188�
"#$%&'()v�s
l�i
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
p

 
� "%�"
�
0���������
� �
>__inference_model_layer_call_and_return_conditional_losses_927�
"#$%&'()t�q
j�g
]�Z
+�(
input_2�����������
+�(
input_3�����������
p 

 
� "%�"
�
0���������
� �
>__inference_model_layer_call_and_return_conditional_losses_964�
"#$%&'()t�q
j�g
]�Z
+�(
input_2�����������
+�(
input_3�����������
p

 
� "%�"
�
0���������
� �
$__inference_model_layer_call_fn_1018�
"#$%&'()v�s
l�i
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
p 

 
� "�����������
$__inference_model_layer_call_fn_1044�
"#$%&'()v�s
l�i
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
p

 
� "�����������
#__inference_model_layer_call_fn_745�
"#$%&'()t�q
j�g
]�Z
+�(
input_2�����������
+�(
input_3�����������
p 

 
� "�����������
#__inference_model_layer_call_fn_890�
"#$%&'()t�q
j�g
]�Z
+�(
input_2�����������
+�(
input_3�����������
p

 
� "�����������
?__inference_preds_layer_call_and_return_conditional_losses_1350]0�-
&�#
!�
inputs���������� 
� "%�"
�
0���������
� x
$__inference_preds_layer_call_fn_1339P0�-
&�#
!�
inputs���������� 
� "�����������
!__inference_signature_wrapper_992�
"#$%&'()}�z
� 
s�p
6
input_2+�(
input_2�����������
6
input_3+�(
input_3�����������"-�*
(
preds�
preds���������