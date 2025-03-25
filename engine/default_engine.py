import os
import cv2
import sys
import torch
import numpy as np
import detectron2.data.transforms as T

from yacs.config import CfgNode
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.structures import Instances
from detectron2.evaluation import COCOEvaluator
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import build_detection_test_loader, build_detection_train_loader, MetadataCatalog

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from engine import BasePredictor
from utils.default_config import _C
from utils.hook import LossEvalHook
from utils.data_utils import build_mapper, load_image, split_image_into_slices, merge_slices


class MaskRCNNTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        mapper = build_mapper(resize=cfg.INPUT.RESIZE, 
                              noise_type=cfg.INPUT.NOISE_TYPE, 
                              noise_param=cfg.INPUT.NOISE_PARAM)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name: str):
        mapper = build_mapper(resize=cfg.INPUT.RESIZE, noise_type='none')
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name: str, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                mapper=build_mapper(resize=self.cfg.INPUT.RESIZE, noise_type='none'))))
        return hooks


class MaskRCNNPredictor(BasePredictor):
    def __call__(self, image_arr: np.ndarray):
        return self._base_call(image_arr)[0]

    def _extract_binary_mask(self, instances: Instances) -> np.ndarray:
        if not instances.has("pred_masks"):
            raise ValueError("The instances object does not have 'pred_masks'.")

        pred_masks = instances.pred_masks.cpu().numpy()
        binary_mask = np.any(pred_masks, axis=0).astype(np.uint8) * 255
        return binary_mask

    def _overlay_mask_on_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask = (mask > 0).astype(np.uint8) * 255
        mask_color = np.zeros_like(image)
        mask_color[:, :, 1] = mask
        overlayed_image = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)
        return overlayed_image

    def _save_vehicle_locations(self, instances: Instances, save_path: str):
        if not instances.has("pred_boxes"):
            raise ValueError("The instances object does not have 'pred_boxes'.")

        boxes = instances.pred_boxes.tensor.cpu().numpy()
        with open(save_path, 'w') as f:
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                f.write(f"Vehicle {idx + 1}: Bounding Box [x1={x1}, y1={y1}, x2={x2}, y2={y2}]\n")

    def get_vehicle_locations(self, image: np.ndarray) -> list:
        """
        Detect vehicles in the given image and return their bounding box locations.

        Args:
            image: A numpy array representing the input image.

        Returns:
            A list of dictionaries containing bounding box coordinates for each detected vehicle.
        """
        predictions = self.__call__(image)

        if 'instances' not in predictions:
            raise ValueError("No 'instances' found in prediction output.")

        instances: Instances = predictions['instances']

        if not instances.has("pred_boxes"):
            raise ValueError("No bounding boxes found in the prediction output.")

        boxes = instances.pred_boxes.tensor.cpu().numpy()
        vehicle_locations = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            vehicle_locations.append({
                "vehicle_id": idx + 1,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            })

        return vehicle_locations

    def inference_on_single_image(
        self, 
        image_file: str, 
        save_dir: str, 
        image_scale: float = 1.0, 
        grid_split: bool = False, 
        split_size: int = None
    ):
        # Create the 'combined_output' folder inside the save_dir
        combined_output_dir = os.path.join(save_dir, "combined_output")
        os.makedirs(combined_output_dir, exist_ok=True)

        if not grid_split:
            img_arr = load_image(image_file)
            pred = self(img_arr)

            # Extract number of vehicles detected
            num_vehicles = len(pred['instances']) if 'instances' in pred else 0

            # Draw predicted instances on the image
            v = Visualizer(img_arr, metadata=self.metadata, scale=image_scale, instance_mode=ColorMode.IMAGE_BW)
            out = v.draw_instance_predictions(pred['instances'].to('cpu'))
            seg_image = out.get_image()[:, :, ::-1]

            # Generate binary mask
            instance_mask = self._extract_binary_mask(pred['instances'])

            # Generate traffic overlay
            traffic_image = self._overlay_mask_on_image(img_arr[:, :, ::-1], instance_mask)

            # Save vehicle locations to a text file
            vehicle_locations_path = os.path.join(combined_output_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}_vehicle_locations.txt")
            self._save_vehicle_locations(pred['instances'], vehicle_locations_path)

        else:
            raise NotImplementedError("Grid split mode is not supported for combined output.")

        # Resize all images to the same dimensions
        image_size = (400, 400)
        resized_input = cv2.resize(img_arr, image_size)
        resized_seg = cv2.resize(seg_image, image_size)
        resized_mask = cv2.cvtColor(cv2.resize(instance_mask, image_size), cv2.COLOR_GRAY2BGR)
        resized_traffic = cv2.resize(traffic_image, image_size)

        # Create labels for each image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        label_color = (0, 0, 0)

        def add_label(image, label):
            labeled_image = cv2.copyMakeBorder(image, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = (labeled_image.shape[1] - text_size[0]) // 2
            cv2.putText(labeled_image, label, (text_x, 30), font, font_scale, label_color, thickness)
            return labeled_image

        # Add labels, including vehicle count
        labeled_input = add_label(resized_input, "Input Image")
        labeled_seg = add_label(resized_seg, f"Segmentation Image (Vehicles: {num_vehicles})")
        labeled_mask = add_label(resized_mask, f"Binary Mask Image (Vehicles: {num_vehicles})")
        labeled_traffic = add_label(resized_traffic, f"Traffic Overlay Image (Vehicles: {num_vehicles})")

        # Add subtle borders to each image
        border_thickness = 5
        bordered_input = cv2.copyMakeBorder(labeled_input, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=(200, 200, 200))
        bordered_seg = cv2.copyMakeBorder(labeled_seg, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=(200, 200, 200))
        bordered_mask = cv2.copyMakeBorder(labeled_mask, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=(200, 200, 200))
        bordered_traffic = cv2.copyMakeBorder(labeled_traffic, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=(200, 200, 200))

        # Combine images into a 2x2 grid
        spacing = 20
        top_row = np.hstack((bordered_input, np.full((bordered_input.shape[0], spacing, 3), 255, dtype=np.uint8), bordered_seg))
        bottom_row = np.hstack((bordered_mask, np.full((bordered_mask.shape[0], spacing, 3), 255, dtype=np.uint8), bordered_traffic))
        grid = np.vstack((top_row, np.full((spacing, top_row.shape[1], 3), 255, dtype=np.uint8), bottom_row))

        # Add a title bar
        title = "Traffic Analysis Output"
        title_height = 60
        title_bar = np.full((title_height, grid.shape[1], 3), 255, dtype=np.uint8)
        text_size = cv2.getTextSize(title, font, 1.2, thickness)[0]
        text_x = (title_bar.shape[1] - text_size[0]) // 2
        text_y = (title_bar.shape[0] + text_size[1]) // 2
        cv2.putText(title_bar, title, (text_x, text_y), font, 1.2, label_color, thickness)

        # Combine title and grid
        final_image = np.vstack((title_bar, grid))

        # Generate a unique filename based on the input image name
        base_name = os.path.basename(image_file)
        name_without_ext, ext = os.path.splitext(base_name)
        combined_path = os.path.join(combined_output_dir, f"{name_without_ext}_combined{ext}")

        # Save combined image
        cv2.imwrite(combined_path, final_image)
        print(f"Combined image saved at: {combined_path}")
