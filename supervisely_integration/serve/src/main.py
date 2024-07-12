from pathlib import Path
import supervisely as sly
from supervisely.nn.prediction_dto import PredictionPoint
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Literal, Union
from cotracker.predictor import CoTrackerPredictor
import torch
import numpy as np

load_dotenv(os.path.expanduser("supervisely.env"))
load_dotenv("./supervisely_integration/serve/debug.env")

os.environ["SMART_CACHE_TTL"] = str(5 * 60)
os.environ["SMART_CACHE_SIZE"] = str(512)
checkpoints_dir = "/checkpoints/"
checkpoint_name = os.environ.get("modal.state.modelName", "cotracker_stride_4_wind_8.pth")


class CoTrackerModel(sly.nn.inference.PointTracking):
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        self.model = CoTrackerPredictor(checkpoint=checkpoint_path)
        self.device = torch.device(device)
        self.model = self.model.to(device)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        rgb_images: List[np.ndarray],
        settings: Dict[str, Any],
        start_object: PredictionPoint,
    ) -> List[PredictionPoint]:
        # cotracker fails to track short sequences, so it is necessary to lengthen them by duplicating last frame
        lengthened = False
        if len(rgb_images) < 11:
            lengthened = True
            original_length = len(rgb_images) - 1  # do not include input frame
            while len(rgb_images) < 11:
                rgb_images.append(rgb_images[-1])
        # disable gradient calculation
        torch.set_grad_enabled(False)
        class_name = start_object.class_name
        input_video = torch.from_numpy(np.array(rgb_images)).permute(0, 3, 1, 2)[None].float()
        input_video = input_video.to(self.device)
        query = torch.tensor([[0, start_object.col, start_object.row]]).float()
        query = query.to(self.device)
        pred_tracks, pred_visibility = self.model(input_video, queries=query[None])
        pred_tracks = pred_tracks.squeeze().cpu()[1:]
        if lengthened:
            pred_tracks = pred_tracks[:original_length]
        pred_points = [
            PredictionPoint(class_name, col=float(track[0]), row=float(track[1]))
            for track in pred_tracks
        ]
        return pred_points
    
    def predict_batch(
        self,
        rgb_images: List[np.ndarray],
        settings: Dict[str, Any],
        start_object: Union[PredictionPoint, List[PredictionPoint]],
    ) -> List[PredictionPoint]:
        # cotracker fails to track short sequences, so it is necessary to lengthen them by duplicating last frame
        if len(rgb_images) == 0:
            return []
        if isinstance(rgb_images[0], str):
            rgb_images = [sly.image.read(path) for path in rgb_images]
        lengthened = False
        if len(rgb_images) < 11:
            lengthened = True
            original_length = len(rgb_images) - 1  # do not include input frame
            while len(rgb_images) < 11:
                rgb_images.append(rgb_images[-1])
        # disable gradient calculation
        torch.set_grad_enabled(False)
        if not isinstance(start_object, list):
            start_object = [start_object]
        class_names = [obj.class_name for obj in start_object]
        input_video = torch.from_numpy(np.array(rgb_images)).permute(0, 3, 1, 2)[None].float()
        input_video = input_video.to(self.device)
        query = torch.tensor([[0, obj.col, obj.row] for obj in start_object]).float()
        query = query.to(self.device)
        pred_tracks, pred_visibility = self.model(input_video, queries=query[None])
        pred_tracks: torch.Tensor
        pred_tracks = pred_tracks.cpu()[0][1:]
        if lengthened:
            pred_tracks = pred_tracks[:original_length]
        pred_points = [
            [
                PredictionPoint(class_name, col=float(point[0]), row=float(point[1]))
                for point, class_name in zip(frame_track, class_names)
            ] for frame_track in pred_tracks
        ]
        return pred_points


model = CoTrackerModel(
    custom_inference_settings=str(Path(__file__).parents[1].joinpath("model_settings.yaml").resolve())
)
model.serve()
