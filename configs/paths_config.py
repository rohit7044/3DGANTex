from pathlib import Path

dataset_paths = {
	'celeba_train': Path(''),
	'celeba_test': Path(''),

	'ffhq': Path(''),
	'ffhq_unaligned': Path('')
}

model_paths = {
	# models for backbones and losses
	'ir_se50': Path('/home/ci3d/repository/stylegan3-editing/pretrained_models/model_ir_se50.pth'),
	# stylegan3 generators
	'stylegan3_ffhq': Path('/home/ci3d/repository/stylegan3-editing/pretrained_models/stylegan3-r-ffhq-1024x1024.pkl'),
	'stylegan3_ffhq_pt': Path('/home/ci3d/repository/stylegan3-editing/pretrained_models/sg3-r-ffhq-1024.pt'),
	'stylegan3_ffhq_unaligned': Path('/home/ci3d/repository/stylegan3-editing/pretrained_models/stylegan3-r-ffhqu-1024x1024.pkl'),
	'stylegan3_ffhq_unaligned_pt': Path('/home/ci3d/repository/stylegan3-editing/pretrained_models/sg3-r-ffhqu-1024.pt'),
	# model for face alignment
	'shape_predictor': Path('/home/ci3d/repository/stylegan3-editing/pretrained_models/shape_predictor_68_face_landmarks.dat'),
	# models for ID similarity computation
	'curricular_face': Path('/home/ci3d/repository/stylegan3-editing/pretrained_models/CurricularFace_Backbone.pth'),
	'mtcnn_pnet': Path('/home/ci3d/repository/stylegan3-editing/pretrained_models/mtcnn/pnet.npy'),
	'mtcnn_rnet': Path('/home/ci3d/repository/stylegan3-editing/pretrained_models/mtcnn/rnet.npy'),
	'mtcnn_onet': Path('/home/ci3d/repository/stylegan3-editing/pretrained_models/mtcnn/onet.npy'),
	# classifiers used for interfacegan training
	'age_estimator': Path('/home/ci3d/repository/stylegan3-editing/pretrained_models/dex_age_classifier.pth'),
	'pose_estimator': Path('/home/ci3d/repository/stylegan3-editing/pretrained_models/hopenet_robust_alpha1.pkl')
}

styleclip_directions = {
	"ffhq": {
		'delta_i_c': Path('/home/ci3d/repository/stylegan3-editing/editing/styleclip_global_directions/sg3-r-ffhq-1024/delta_i_c.npy'),
		's_statistics': Path('/home/ci3d/repository/stylegan3-editing/editing/styleclip_global_directions/sg3-r-ffhq-1024/s_stats'),
	},
	'templates': Path('/home/ci3d/repository/stylegan3-editing/editing/styleclip_global_directions/templates.txt')
}

interfacegan_aligned_edit_paths = {
	'age': Path('/home/ci3d/repository/stylegan3-editing/editing/interfacegan/boundaries/ffhq/age_boundary.npy'),
	'smile': Path('/home/ci3d/repository/stylegan3-editing/editing/interfacegan/boundaries/ffhq/Smiling_boundary.npy'),
	'pose': Path('/home/ci3d/repository/stylegan3-editing/editing/interfacegan/boundaries/ffhq/pose_boundary.npy'),
	'Male': Path('/home/ci3d/repository/stylegan3-editing/editing/interfacegan/boundaries/ffhq/Male_boundary.npy'),
}

interfacegan_unaligned_edit_paths = {
	'age': Path('editing/interfacegan/boundaries/ffhqu/age_boundary.npy'),
	'smile': Path('editing/interfacegan/boundaries/ffhqu/Smiling_boundary.npy'),
	'pose': Path('editing/interfacegan/boundaries/ffhqu/pose_boundary.npy'),
	'Male': Path('editing/interfacegan/boundaries/ffhqu/Male_boundary.npy'),
}
