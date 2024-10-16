from pathlib import Path

# Get the current working directory
current_directory = Path.cwd()

# Traverse up the directory structure until we find the 3D-GANTex directory
base_path = current_directory
while base_path.name != '3D-GANTex' and base_path.parent != base_path:
    base_path = base_path.parent

# If we didn't find the directory, you might want to handle that case
if base_path.name != '3D-GANTex':
    raise FileNotFoundError("3D-GANTex directory not found in the current path")

dataset_paths = {
    'celeba_train': base_path / '',
    'celeba_test': base_path / '',
    'ffhq': base_path / '',
    'ffhq_unaligned': base_path / ''
}

model_paths = {
    # models for backbones and losses
    'ir_se50': base_path / 'pretrained_models/model_ir_se50.pth',
    # stylegan3 generators
    'stylegan3_ffhq': base_path / 'pretrained_models/stylegan3-r-ffhq-1024x1024.pkl',
    'stylegan3_ffhq_pt': base_path / 'pretrained_models/sg3-r-ffhq-1024.pt',
    'stylegan3_ffhq_unaligned': base_path / 'pretrained_models/stylegan3-r-ffhqu-1024x1024.pkl',
    'stylegan3_ffhq_unaligned_pt': base_path / 'pretrained_models/sg3-r-ffhqu-1024.pt',
    # model for face alignment
    'shape_predictor': base_path / 'pretrained_models/shape_predictor_68_face_landmarks.dat',
    # models for ID similarity computation
    'curricular_face': base_path / 'pretrained_models/CurricularFace_Backbone.pth',
    'mtcnn_pnet': base_path / 'pretrained_models/mtcnn/pnet.npy',
    'mtcnn_rnet': base_path / 'pretrained_models/mtcnn/rnet.npy',
    'mtcnn_onet': base_path / 'pretrained_models/mtcnn/onet.npy',
    # classifiers used for interfacegan training
    'age_estimator': base_path / 'pretrained_models/dex_age_classifier.pth',
    'pose_estimator': base_path / 'pretrained_models/hopenet_robust_alpha1.pkl'
}

styleclip_directions = {
    "ffhq": {
        'delta_i_c': base_path / 'editing/styleclip_global_directions/sg3-r-ffhq-1024/delta_i_c.npy',
        's_statistics': base_path / 'editing/styleclip_global_directions/sg3-r-ffhq-1024/s_stats',
    },
    'templates': base_path / 'editing/styleclip_global_directions/templates.txt'
}

interfacegan_aligned_edit_paths = {
    'age': base_path / 'editing/interfacegan/boundaries/ffhq/age_boundary.npy',
    'smile': base_path / 'editing/interfacegan/boundaries/ffhq/Smiling_boundary.npy',
    'pose': base_path / 'editing/interfacegan/boundaries/ffhq/pose_boundary.npy',
    'Male': base_path / 'editing/interfacegan/boundaries/ffhq/Male_boundary.npy',
}

interfacegan_unaligned_edit_paths = {
    'age': base_path / 'editing/interfacegan/boundaries/ffhqu/age_boundary.npy',
    'smile': base_path / 'editing/interfacegan/boundaries/ffhqu/Smiling_boundary.npy',
    'pose': base_path / 'editing/interfacegan/boundaries/ffhqu/pose_boundary.npy',
    'Male': base_path / 'editing/interfacegan/boundaries/ffhqu/Male_boundary.npy',
}
