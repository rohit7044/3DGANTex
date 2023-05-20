import time
import numpy as np
from PIL import Image
import torch
import dlib
import torchvision.transforms as transforms
import yaml
from models.stylegan3.model import GeneratorType
from utils.common import tensor2im
from utils.inference_utils import run_on_batch, load_encoder, get_average_image
from utils.alignment_utils import align_face, crop_face, get_stylegan_transform
from editing.interfacegan.face_editor import FaceEditor
from ThreeDDFA_utils.uv import uv_tex
from ThreeDDFA_utils.serialization import ser_to_obj
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA.TDDFA_ONNX import TDDFA_ONNX


# Prepare the Data
experiment_type = 'restyle_e4e_ffhq' # can choose between e4e and pSp encoding
# Load the weights
pSp_model_path = "./pretrained_models/restyle_pSp_ffhq.pt"
e4e_model_path = "./pretrained_models/restyle_e4e_ffhq.pt"
shape_predictor_path = "./pretrained_models/shape_predictor_68_face_landmarks.dat"
cfg = yaml.load(open('ThreeDDFA_configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Inversion iteration Higher means getting closer. But sometimes 1 iteration produce good results
n_iters_per_batch = 3
# Input image
face_img_path = '/home/ci3d/repository/3D-GANTex/input_data/00012.png' # Change the file name here
pose_img_path = f'output_data/{face_img_path.split("/")[-1].replace(".png", "")}_pose' + '.png' # Change the file name here
uv_tex_path = f'output_data/{face_img_path.split("/")[-1].replace(".png", "")}_uv_tex' + '.png'# Change the file name here
obj_tex_path = f'output_data/{face_img_path.split("/")[-1].replace(".png", "")}_obj' + '.obj'
edit_direction = 'pose'
# Range of the pose
min_value = -3
max_value = 5
# out of the range above, take the index which will generate the frontal face
frontal_face_index = -1
# Define Inference Parameters
EXPERIMENT_DATA_ARGS = {
    "restyle_pSp_ffhq": {
        "model_path": pSp_model_path,
        "image_path": face_img_path,
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "restyle_e4e_ffhq": {
        "model_path": e4e_model_path,
        "image_path": face_img_path,
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]



def run_alignment(input_img):
    predictor = dlib.shape_predictor(shape_predictor_path)
    detector = dlib.get_frontal_face_detector()
    print("Aligning image...")
    aligned_image = align_face(input_img, detector=detector, predictor=predictor)
    print(f"Finished aligning image: {image_path}")
    return aligned_image

def crop_image(input_img):
    predictor = dlib.shape_predictor(shape_predictor_path)
    detector = dlib.get_frontal_face_detector()
    print("Cropping image...")
    cropped_image = crop_face(input_img, detector=detector, predictor=predictor)
    print(f"Finished cropping image: {image_path}")
    return cropped_image

def compute_transforms(aligned_img, cropped_img):
    predictor = dlib.shape_predictor(shape_predictor_path)
    detector = dlib.get_frontal_face_detector()
    print("Computing landmarks-based transforms...")
    res = get_stylegan_transform(cropped_img, aligned_img, detector, predictor)
    print("Done!")
    if res is None:
        print(f"Failed computing transforms on: compute transform")
        return
    else:
        rotation_angle, translation, transform, inverse_transform = res
        return inverse_transform

# load the encoder
model_path = EXPERIMENT_ARGS['model_path']
net, opts = load_encoder(checkpoint_path=model_path)
# pprint.pprint(dataclasses.asdict(opts))
# Show the image
image_path = str(EXPERIMENT_DATA_ARGS[experiment_type]["image_path"])
original_image = Image.open(image_path).convert("RGB")

# Get aligned and cropped image
aligned_image = run_alignment(original_image)
cropped_image = crop_image(original_image)
# cropped_image.show()

# Compute landmark based transform
landmarks_transform = compute_transforms(aligned_image,cropped_image)

#perform inversion
n_iters_per_batch = 3
opts.n_iters_per_batch = n_iters_per_batch
opts.resize_outputs = False  # generate outputs at full resolution

img_transforms = EXPERIMENT_ARGS['transform']
transformed_image = img_transforms(original_image)

avg_image = get_average_image(net)

with torch.no_grad():
    tic = time.time()
    result_batch, result_latents = run_on_batch(inputs=transformed_image.unsqueeze(0).cuda().float(),
                                                net=net,
                                                opts=opts,
                                                avg_image=avg_image)
    toc = time.time()
    print('Inference took {:.4f} seconds.'.format(toc - tic))

result_tensors = result_batch[0]
inversed_img = tensor2im(result_tensors[-1])
inversed_img.save(pose_img_path)

# Latent Space Editing using InterFaceGAN
editor = FaceEditor(stylegan_generator=net.decoder, generator_type=GeneratorType.ALIGNED)


print(f"Performing edit for {edit_direction}...")
input_latent = torch.from_numpy(result_latents[0][-1]).unsqueeze(0).cuda()
edit_images, edit_latents = editor.edit(latents=input_latent,
                                        direction=edit_direction,
                                        factor_range=(min_value, max_value),
                                        user_transforms=landmarks_transform,
                                        apply_user_transformations=True)
print("Done!")

# Pose Editing
edit_images = [image[0] for image in edit_images]
res = np.array(edit_images[0].resize((512, 512)))
frontal_face_img = np.asarray(edit_images[frontal_face_index]) # which image to choose
# show pose images
for image in edit_images[1:]:
    res = np.concatenate([res, image.resize((512, 512))], axis=1)
pose_img = Image.fromarray(res).convert("RGB")
pose_img.show()
pose_img.save(pose_img_path)

# Generate UV Map using 3DDFA_V2

frontal_face_img_rgb = frontal_face_img[:, :, ::-1]

# Init FaceBoxes and TDDFA, recommend using onnx flag
face_boxes = FaceBoxes_ONNX()
tddfa = TDDFA_ONNX(**cfg)

# Detect Face
boxes = face_boxes(frontal_face_img_rgb)
param_lst, roi_box_lst = tddfa(frontal_face_img_rgb, boxes)
ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

#UV_texture
uv_tex(frontal_face_img_rgb,ver_lst,tddfa.tri,wfp= uv_tex_path )
ser_to_obj(frontal_face_img_rgb, ver_lst, tddfa.tri,height = 1024, wfp=obj_tex_path)

# show the .obj file
import open3d as o3d

# Read the OBJ file
mesh = o3d.io.read_triangle_mesh(obj_tex_path)

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])

# Calculate SSIM
# compute(cropped_image,inversed_img)


