import os
import logging
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)


def _get_cityscapes_aug_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    cities = os.listdir(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in os.listdir(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            suffix = "rgb_anon.png"
            assert basename.endswith(suffix), basename
            basename = basename[: -len(suffix)]

            label_file = os.path.join(city_gt_dir, basename + "gt_labelIds.png")

            files.append((image_file, label_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert os.path.isfile(f), f
    return files


def load_cityscapes_aug_semantic(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, label_file in _get_cityscapes_aug_files(image_dir, gt_dir):
        label_file = label_file.replace("labelIds", "labelTrainIds")

        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": 1024,
                "width": 2048,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    return ret


def register_all_cityscapes_aug(root):
    domains = ["night"]
    splits = ["train"]

    for domain in domains:
        for split in splits:
            dataset_name = f"cityscapes_aug_{domain}_fine_sem_seg_{split}"
            image_dir = f"cityscapes/aug_{domain}/{split}/"
            gt_dir = f"cityscapes/gtFine/{split}/"

            image_dir = os.path.join(root, image_dir)
            gt_dir = os.path.join(root, gt_dir)

            meta = _get_builtin_metadata("cityscapes")
            MetadataCatalog.get(dataset_name).set(
                evaluator_type="cityscapes_aug_sem_seg",
                ignore_label=255,
                image_dir=image_dir,
                gt_dir=gt_dir,
                **meta,
            )

            DatasetCatalog.register(
                dataset_name,
                lambda x=image_dir, y=gt_dir: load_cityscapes_aug_semantic(x, y),
            )


dataset_path = os.environ["DETECTRON2_DATASETS"]
if dataset_path is None:
    dataset_path = "datasets"

register_all_cityscapes_aug(dataset_path)
