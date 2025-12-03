from typing import Dict, List, Optional, Union

from dataset_tools.templates import (
    AnnotationType,
    Category,
    CVTask,
    Domain,
    Industry,
    License,
    Research,
)

##################################
# * Before uploading to instance #
##################################
PROJECT_NAME: str = "TrashCan 1.0"
PROJECT_NAME_FULL: str = (
    "TrashCan 1.0: An Instance-Segmentation Labeled Dataset of Trash Observations"
)
HIDE_DATASET = False  # set False when 100% sure about repo quality

##################################
# * After uploading to instance ##
##################################
LICENSE: License = License.Custom(
    source_url="https://conservancy.umn.edu/bitstream/handle/11299/214865/LICENSE.txt?sequence=2&isAllowed=y"
)
APPLICATIONS: List[Union[Industry, Domain, Research]] = [
    Industry.WasteRecycling(),
    Industry.Marine(),
    Industry.Robotics(),
]
CATEGORY: Category = Category.Environmental(extra=Category.Robotics())

CV_TASKS: List[CVTask] = [
    CVTask.InstanceSegmentation(),
    CVTask.SemanticSegmentation(),
    CVTask.ObjectDetection(),
]
ANNOTATION_TYPES: List[AnnotationType] = [
    AnnotationType.InstanceSegmentation(),
    AnnotationType.ObjectDetection(),
]

RELEASE_DATE: Optional[str] = "2020-07-23"  # e.g. "YYYY-MM-DD"
if RELEASE_DATE is None:
    RELEASE_YEAR: int = None

HOMEPAGE_URL: str = "https://conservancy.umn.edu/handle/11299/214865"
# e.g. "https://some.com/dataset/homepage"

PREVIEW_IMAGE_ID: int = 14431163
# This should be filled AFTER uploading images to instance, just ID of any image.

GITHUB_URL: str = "https://github.com/dataset-ninja/trash-can"
# URL to GitHub repo on dataset ninja (e.g. "https://github.com/dataset-ninja/some-dataset")

##################################
### * Optional after uploading ###
##################################
DOWNLOAD_ORIGINAL_URL: Optional[Union[str, dict]] = (
    "https://conservancy.umn.edu/bitstream/handle/11299/214865/dataset.zip?sequence=12&isAllowed=y"
)
# Optional link for downloading original dataset (e.g. "https://some.com/dataset/download")

CLASS2COLOR: Optional[Dict[str, List[str]]] = {
    "rov": [230, 25, 75],
    "plant": [60, 180, 75],
    "animal fish": [255, 225, 25],
    "animal starfish": [0, 130, 200],
    "animal shells": [245, 130, 48],
    "animal crab": [145, 30, 180],
    "animal eel": [70, 240, 240],
    "animal etc": [240, 50, 230],
    "trash etc": [210, 245, 60],
    "trash fabric": [250, 190, 212],
    "trash fishing gear": [0, 128, 128],
    "trash metal": [220, 190, 255],
    "trash paper": [170, 110, 40],
    "trash plastic": [255, 250, 200],
    "trash rubber": [128, 0, 0],
    "trash wood": [170, 255, 195],
    "trash unknown instance": [128, 128, 0],
    "trash branch": [255, 215, 180],
    "trash wreckage": [0, 0, 128],
    "trash tarp": [200, 54, 128],
    "trash rope": [40, 200, 0],
    "trash net": [235, 155, 40],
}

# If specific colors for classes are needed, fill this dict (e.g. {"class1": [255, 0, 0], "class2": [0, 255, 0]})

# If you have more than the one paper, put the most relatable link as the first element of the list
# Use dict key to specify name for a button
PAPER: Optional[Union[str, List[str], Dict[str, str]]] = "https://arxiv.org/pdf/2007.08097"
BLOGPOST: Optional[Union[str, List[str], Dict[str, str]]] = None
REPOSITORY: Optional[Union[str, List[str], Dict[str, str]]] = None

CITATION_URL: Optional[str] = None
AUTHORS: Optional[List[str]] = ["Hong Jungseok", "Fulton Michael", "Sattar Junaed"]
AUTHORS_CONTACTS: Optional[List[str]] = ["jungseok@umn.edu", "fulto081@umn.edu", "junaed@umn.edu"]

ORGANIZATION_NAME: Optional[Union[str, List[str]]] = "University of Minnesota Twin Cities, USA"
ORGANIZATION_URL: Optional[Union[str, List[str]]] = "https://twin-cities.umn.edu/"

# Set '__PRETEXT__' or '__POSTTEXT__' as a key with string value to add custom text. e.g. SLYTAGSPLIT = {'__POSTTEXT__':'some text}
SLYTAGSPLIT: Optional[Dict[str, Union[List[str], str]]] = {
    "__POSTTEXT__": "Additionally, every image marked with its ***video id*** tag"
}
TAGS: Optional[List[str]] = None


SECTION_EXPLORE_CUSTOM_DATASETS: Optional[List[str]] = None

##################################
###### ? Checks. Do not edit #####
##################################


def check_names():
    fields_before_upload = [PROJECT_NAME]  # PROJECT_NAME_FULL
    if any([field is None for field in fields_before_upload]):
        raise ValueError("Please fill all fields in settings.py before uploading to instance.")


def get_settings():
    if RELEASE_DATE is not None:
        global RELEASE_YEAR
        RELEASE_YEAR = int(RELEASE_DATE.split("-")[0])

    settings = {
        "project_name": PROJECT_NAME,
        "project_name_full": PROJECT_NAME_FULL or PROJECT_NAME,
        "hide_dataset": HIDE_DATASET,
        "license": LICENSE,
        "applications": APPLICATIONS,
        "category": CATEGORY,
        "cv_tasks": CV_TASKS,
        "annotation_types": ANNOTATION_TYPES,
        "release_year": RELEASE_YEAR,
        "homepage_url": HOMEPAGE_URL,
        "preview_image_id": PREVIEW_IMAGE_ID,
        "github_url": GITHUB_URL,
    }

    if any([field is None for field in settings.values()]):
        raise ValueError("Please fill all fields in settings.py after uploading to instance.")

    settings["release_date"] = RELEASE_DATE
    settings["download_original_url"] = DOWNLOAD_ORIGINAL_URL
    settings["class2color"] = CLASS2COLOR
    settings["paper"] = PAPER
    settings["blog"] = BLOGPOST
    settings["repository"] = REPOSITORY
    settings["citation_url"] = CITATION_URL
    settings["authors"] = AUTHORS
    settings["authors_contacts"] = AUTHORS_CONTACTS
    settings["organization_name"] = ORGANIZATION_NAME
    settings["organization_url"] = ORGANIZATION_URL
    settings["slytagsplit"] = SLYTAGSPLIT
    settings["tags"] = TAGS

    settings["explore_datasets"] = SECTION_EXPLORE_CUSTOM_DATASETS

    return settings
