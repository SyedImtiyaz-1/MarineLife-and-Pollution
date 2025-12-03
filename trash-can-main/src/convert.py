import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import get_file_name, get_file_name_with_ext
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    batch_size = 30

    ds_name_to_splits = {
        "materials train": (
            "/home/alex/DATASETS/TODO/TrashCan/dataset/material_version/train",
            "/home/alex/DATASETS/TODO/TrashCan/dataset/material_version/instances_train_trashcan.json",
        ),
        "materials val": (
            "/home/alex/DATASETS/TODO/TrashCan/dataset/material_version/val",
            "/home/alex/DATASETS/TODO/TrashCan/dataset/material_version/instances_val_trashcan.json",
        ),
        "instance train": (
            "/home/alex/DATASETS/TODO/TrashCan/dataset/instance_version/train",
            "/home/alex/DATASETS/TODO/TrashCan/dataset/instance_version/instances_train_trashcan.json",
        ),
        "instance val": (
            "/home/alex/DATASETS/TODO/TrashCan/dataset/instance_version/val",
            "/home/alex/DATASETS/TODO/TrashCan/dataset/instance_version/instances_val_trashcan.json",
        ),
    }

    def create_ann(image_path):
        labels = []
        tags = []

        image_name = get_file_name_with_ext(image_path)
        img_height = image_name_to_shape[image_name][0]
        img_wight = image_name_to_shape[image_name][1]

        video_value = int(image_name.split("_")[1])
        video_tag = sly.Tag(video_id_meta, value=video_value)
        tags.append(video_tag)

        ann_data = image_name_to_ann_data[image_name]
        for curr_ann_data in ann_data:
            category_id = curr_ann_data[0]
            obj_class = idx_to_class[category_id]
            polygons_coords = curr_ann_data[1]
            for coords in polygons_coords:
                exterior = []
                for i in range(0, len(coords), 2):
                    exterior.append([int(coords[i + 1]), int(coords[i])])
                if len(exterior) < 3:
                    continue
                poligon = sly.Polygon(exterior)
                if poligon.area > 30:
                    label_poly = sly.Label(poligon, obj_class)
                    labels.append(label_poly)

            bbox_coord = curr_ann_data[2]
            rectangle = sly.Rectangle(
                top=int(bbox_coord[1]),
                left=int(bbox_coord[0]),
                bottom=int(bbox_coord[1] + bbox_coord[3]),
                right=int(bbox_coord[0] + bbox_coord[2]),
            )
            label_rectangle = sly.Label(rectangle, obj_class)
            labels.append(label_rectangle)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    idx_to_class = {
        1: sly.ObjClass("rov", sly.AnyGeometry),
        2: sly.ObjClass("plant", sly.AnyGeometry),
        3: sly.ObjClass("animal fish", sly.AnyGeometry),
        4: sly.ObjClass("animal starfish", sly.AnyGeometry),
        5: sly.ObjClass("animal shells", sly.AnyGeometry),
        6: sly.ObjClass("animal crab", sly.AnyGeometry),
        7: sly.ObjClass("animal eel", sly.AnyGeometry),
        8: sly.ObjClass("animal etc", sly.AnyGeometry),
        9: sly.ObjClass("trash etc", sly.AnyGeometry),
        10: sly.ObjClass("trash fabric", sly.AnyGeometry),
        11: sly.ObjClass("trash fishing gear", sly.AnyGeometry),
        12: sly.ObjClass("trash metal", sly.AnyGeometry),
        13: sly.ObjClass("trash paper", sly.AnyGeometry),
        14: sly.ObjClass("trash plastic", sly.AnyGeometry),
        15: sly.ObjClass("trash rubber", sly.AnyGeometry),
        16: sly.ObjClass("trash wood", sly.AnyGeometry),
        17: sly.ObjClass("trash unknown instance", sly.AnyGeometry),
        18: sly.ObjClass("trash branch", sly.AnyGeometry),
        19: sly.ObjClass("trash wreckage", sly.AnyGeometry),
        20: sly.ObjClass("trash tarp", sly.AnyGeometry),
        21: sly.ObjClass("trash rope", sly.AnyGeometry),
        22: sly.ObjClass("trash net", sly.AnyGeometry),
    }

    video_id_meta = sly.TagMeta("video id", sly.TagValueType.ANY_NUMBER)

    meta = sly.ProjectMeta(
        obj_classes=list(idx_to_class.values()),
        tag_metas=[video_id_meta],
    )

    api.project.update_meta(project.id, meta.to_json())

    for ds_name, ds_data in ds_name_to_splits.items():

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)
        image_id_to_name = {}
        image_name_to_ann_data = defaultdict(list)
        image_name_to_shape = {}

        images_path, ann_json = ds_data
        images_names = os.listdir(images_path)

        ann = load_json_file(ann_json)

        for curr_image_info in ann["images"]:
            image_id_to_name[curr_image_info["id"]] = curr_image_info["file_name"]
            image_name_to_shape[curr_image_info["file_name"]] = (
                curr_image_info["height"],
                curr_image_info["width"],
            )

        for curr_ann_data in ann["annotations"]:
            image_id = curr_ann_data["image_id"]
            image_name_to_ann_data[image_id_to_name[image_id]].append(
                [curr_ann_data["category_id"], curr_ann_data["segmentation"], curr_ann_data["bbox"]]
            )

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(images_path, image_path) for image_path in img_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))

    return project
