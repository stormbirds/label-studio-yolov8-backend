import os, sys, shutil, glob, random
import torch
import sqlite3
# from dotenv import load_dotenv
from PIL import Image
import yaml
import logging

import label_studio_sdk

from label_studio_ml import model
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, is_skipped

from utils.io import  get_data_dir
from ultralytics import YOLO

model.LABEL_STUDIO_ML_BACKEND_V2_DEFAULT = True

current_dir = os.path.dirname(os.path.abspath(__file__)) #! Other method didnt work

DEVICE = '0' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 512
MIN_NEW_LABLES = 1
MIN_TRAINING_LABLES = 10

DEFAULT_YOLO_MODEL = 'yolov8m.pt'
################### From other file
IMG_DATA = os.path.join('Data', 'images')
LABEL_DATA = os.path.join('Data', 'labels')
LABEL_STUDIO_DATA_PATH = rf"{os.environ['USERPROFILE']}\AppData\Local\label-studio\label-studio" if sys.platform == "win32" else f"{os.path.expanduser('~')}/.local/share/label-studio" 
TRAINED_WEIGHTS = os.path.join('Models', 'best.pt')#TODO: glob for versions and save location for weights after training
# LABEL_STUDIO_DATA_PATH = "E:\docker\label-studio\data"
LABEL_STUDIO_DATA_PATH = os.environ['LABEL_STUDIO_DB_PATH']
with sqlite3.connect(os.path.join(LABEL_STUDIO_DATA_PATH, "label_studio.sqlite3"), check_same_thread=False) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT key FROM authtoken_token")
    LABEL_STUDIO_ACCESS_TOKEN = cursor.fetchone()[0]
    cursor.close()
LABEL_STUDIO_ACCESS_TOKEN = os.environ['LABEL_STUDIO_API_TOKEN']
# LABEL_STUDIO_HOST = "http://192.168.0.108:8080" #! figure out programatic sometime in the future
LABEL_STUDIO_HOST = os.environ['LABEL_STUDIO_BASEURL']
# LS_API_TOKEN = os.environ['LABEL_STUDIO_API_TOKEN']

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class yoloV8backend(LabelStudioMLBase):
    def __init__(self, device=DEVICE, img_size=IMAGE_SIZE, train_output=None, **kwargs):
        super(yoloV8backend, self).__init__(**kwargs)
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.device = device
        self.image_dir = upload_dir
        self.img_size = img_size
        self.label_map = {}
        self.hostname = LABEL_STUDIO_HOST
        self.access_token = LABEL_STUDIO_ACCESS_TOKEN
        # self.project_id = int(os.path.basename(os.path.abspath('.')))
        self.num_epochs=10

        self.labels = []
        self.label_attrs = []

        logger.critical(os.path.abspath("."))
        logger.critical(os.listdir('.'))
        try:
            models = os.listdir("runs/detect/")
        except:
            models = []
            logger.critical("No models found in runs/detect/")

        print(models)
        runs = [int(s[5:]) if s[5:].isdigit() else 0 for s in models]
        idx = max(runs) if runs else ""
        self.run_number = idx if idx != 0 else ''
        self.current_checkpoint = f'runs/detect/train{self.run_number}/weights/best.pt' 
        

        if os.path.isfile(self.current_checkpoint):
            self.model = YOLO(self.current_checkpoint, task = 'detect')
            logger.critical(f"Using exisiting model at: {self.current_checkpoint}")
            self.is_custom=True
        else:
            self.is_custom=False
            self.model = YOLO(DEFAULT_YOLO_MODEL)
            logger.critical(f"Fresh model initialised!")
            logger.critical(f"Could not find existing weights at: {self.current_checkpoint}")            

        self.from_name, self.to_name, self.value = ('true_bbox', 'image', 'image')

    def get_project_id_from_db(self, task_id):
        """从数据库中获取项目ID"""
        project_id = None
        try:
            with sqlite3.connect(os.path.join(LABEL_STUDIO_DATA_PATH, "label_studio.sqlite3"), check_same_thread=False) as conn:
                cursor = conn.cursor()
                # 使用参数化查询以避免SQL注入
                cursor.execute("select project_id from task where id = ?", (task_id,))
                result = cursor.fetchone()
                if result:
                    project_id = result[0]
                cursor.close()
        except Exception as e:
            logging.error(f"Error fetching project ID from DB for task ID {task_id}: {e}")
        return project_id

    def get_project_from_ls(self, project_id):
        """通过Label Studio SDK获取项目"""
        try:
            ls = label_studio_sdk.Client(LABEL_STUDIO_HOST, LABEL_STUDIO_ACCESS_TOKEN)
            project = ls.get_project(id=project_id)
            return project
        except Exception as e:
            logging.error(f"Error fetching project from Label Studio for project ID {project_id}: {e}")
            return None


    def get_project_labels_by_id(self, projectId):
        """
        根据task_id使用LabelStudioSDK获取task详情，然后基于task信息获取到project_id，然后使用project获取对应labels和labels_attrs，并更新到label_map中
        :param task_id:
        :return:
        """

        project = self.get_project_from_ls(projectId)

        # Get correct labels with checking for legacy bradken from_name
        if 'true_bbox' in project.parsed_label_config:
            self.labels = project.parsed_label_config['true_bbox']['labels']
            self.label_attrs = project.parsed_label_config['true_bbox']['labels_attrs']

        elif 'label' in project.parsed_label_config:
            self.labels = project.parsed_label_config['label']['labels']
            self.label_attrs = project.parsed_label_config['label']['labels_attrs']
        else:
            logger.error("!" * 80)
            logger.error("BAD LABEL CONFIG, THIS MUST BE FIXED")
            logger.error("!" * 80)
            return

        if list(self.model.names.values()) != self.labels:
            logger.warning("WARNING! Labels in model do not match the labels in Label Studio!")
            logger.warning(f"\nModel Labels: \n\t{list(self.model.names.values())}")
            logger.warning(f"\nLabel Studio Labels:\n\t{self.labels}")
            logger.warning("Retraining new model")
            self.model = YOLO(DEFAULT_YOLO_MODEL)
            logger.critical(f"Fresh model initialised!")
            self.is_custom = False

        if self.label_attrs:
            for label_name, label_attrs in self.label_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name


    def _get_image_url(self,task):
        image_url = task['data'][self.value]
        return image_url

    def label2idx(self,label):
        #convert label to according index in data.yaml
        for index, string in enumerate(self.labels):
            if label == string:
                return index
        return None
  
    def reset_train_dir(self, dir_path):
        #remove cache file and reset train/val dir
        if os.path.isfile(os.path.join(dir_path,"train.cache")):
            os.remove(os.path.join(dir_path, "train.cache"))
        if os.path.isfile(os.path.join(dir_path,"val.cache")):
            os.remove(os.path.join(dir_path, "val.cache"))

        for dir in os.listdir(dir_path):
            file = os.path.join(dir_path, dir)
            shutil.rmtree(file)
            os.makedirs(file)
            logger.critical(f"DELETED FILE: {file}")

    
    def download_tasks(self, project):
        """
        Download all labeled tasks from project using the Label Studio SDK.
        Read more about SDK here https://labelstud.io/sdk/
        :param project: project ID
        :return:
        """
        ls = label_studio_sdk.Client(LABEL_STUDIO_HOST, LABEL_STUDIO_ACCESS_TOKEN)
        project = ls.get_project(id=project)
        tasks = project.get_labeled_tasks()
        return tasks

    def extract_data_from_tasks(self, tasks):
        """ This converts the tasks into the txt format required for YOLO training"""
        img_labels = []
        current_images = [os.path.basename(image) for image in glob.glob(f"{IMG_DATA}/**/*", recursive=True)]
        new_tasks = []
        if not self.labels :
            project_id = self.get_project_id_from_db(tasks[0]['id'])
            self.get_project_labels_by_id(project_id)
        logger.critical("Finding New Labels")
        for task in tasks:     
            if is_skipped(task):
                continue
            
            # Get image name from tasks         
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url) #!!!!!!!!!!!!!!! can jjust split url and build actual path
            image_name = os.path.basename(image_path)
            
            # Check for auto_bbox without true_bbox
            # Known bug if somone submits auto-annotation too quickly
            # i.e. submitted before result recieved from backend
            # This causes the image to apear like it has a good annotation visually
            # however under the hood it only has 'auto_bbox'
            results=[]
            for annotation in task.get('annotations'):
                if annotation.get('result'):
                   results.extend(annotation.get('result') if annotation.get('result')[0] is not None else [])

            auto_found, true_found = (False, False)

            for result in results:
                if (result['from_name'] =='true_bbox') or (result['from_name'] =='label'):
                    true_found = True
                elif result['from_name'] =='auto_bbox':
                    auto_found = True
                    
            # logger.critical(auto_found and not true_found)
            # logger.critical(results)
            
            if auto_found and not true_found:
                continue
            
            if not image_name in current_images:
                new_tasks.append(task)
                
        if (len(new_tasks) < MIN_NEW_LABLES) or ((len(new_tasks)+len(current_images))<MIN_TRAINING_LABLES):
            logger.critical("Not enough labels")
            return False
        self.num_epochs = len(new_tasks)
        # Shuffle list for random
        random.shuffle(new_tasks)
        
        logger.critical("splitting new tasks")
        for idx, task in enumerate(new_tasks):

            if (idx >= 0.8*len(new_tasks)):
                split = 'val'
            else:
                split = 'train'
                      
            # Get image name from tasks         
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url) #!!!!!!!!!!!!!!! can jjust split url and build actual path
            image_name = os.path.basename(image_path)
            
            filename, _ = os.path.splitext(image_name)
            
            Image.open(image_path).save(os.path.join(IMG_DATA, split, image_name))

            img_labels.append(task['annotations'][0]['result'])

            for annotation in task['annotations']:
                with open(f'{os.path.join(LABEL_DATA, split, filename)}.txt', 'w') as f:
                    for result in annotation['result']:
                        # logger.critical(result)
                        if result['type'] == 'rectanglelabels' and ((result['from_name'] =='true_bbox') or (result['from_name'] =='label')):
                            if result.get('value').get('width') is None:
                                continue
                            bb_width = (result['value']['width']) / 100
                            bb_height = (result['value']['height']) / 100
                            x = (result['value']['x'] / 100 ) + (bb_width/2)
                            y = (result['value']['y'] / 100 ) + (bb_height/2)
                            label = result['value']['rectanglelabels'][0]
                            label_idx = self.label2idx(label) 
                                
                            f.write(f"{label_idx} {x} {y} {bb_width} {bb_height}\n")
        
        return True

    def generate_training_yaml(self):
        """ This generates the yaml file required for YOLO training"""""
        nc = len(self.labels)
        data = {
            'train': os.path.join('images','train'),
            'val': os.path.join('images','val'),
            'names': {i: self.labels[i] for i in range(nc)},
            'nc': nc,
            'path': os.path.join(os.path.abspath("."),"Data"),
        }

        self.yaml_path = os.path.join('Data', 'split.yaml')
        logger.critical(os.path.dirname(os.path.dirname(self.images_train_dir)))
        logger.critical(self.yaml_path)
        with open(self.yaml_path, 'w') as file:
            yaml.dump(data, file)
        
    def fit(self, event, data, batch_size=32, num_epochs=10, **kwargs):
        # TODO: re-write this to use the provided data from above
        logger.critical("Fit function called")
        #! could delete cache filles then randomlly allocate new labels and images to train and valid!!!!!!
        
        # Create training and validation directories for both images and lables if they don't already exist
        for dir_path in [IMG_DATA, LABEL_DATA]:
            # logger.critical(f"!!!!!!!!!!!!!!!!!!!!!{dir_path}!!!!!!!!!!!!!!!!!!!!!!!!")
            train = os.path.join(dir_path, 'train')
            if not os.path.exists(train):
                os.makedirs(train, exist_ok=True)
                
            val = os.path.join(dir_path, 'val')
            if not os.path.exists(val):
                os.makedirs(val, exist_ok=True)
                
        self.images_train_dir = os.path.join(IMG_DATA, 'train')
        self.images_val_dir = os.path.join(IMG_DATA, 'val')

        if "project" in kwargs:
            logger.critical("Downloading Tasks")
            project = kwargs['project']
            tasks = self.download_tasks(project)
        else:
            return
        
        if not self.extract_data_from_tasks(tasks):
            return
        
        logger.critical("Got data from tasks")
        
        logger.critical("Generating YAML")
        self.generate_training_yaml() 

        logger.critical("Training cycle started!")

        self.model.train(data=self.yaml_path, batch = batch_size, imgsz=512, epochs=self.num_epochs)

        logger.critical("done training")
        
    @property
    def model_name(self):
        return f'YOLO Run {self.run_number}:{self.device}'
    
    def predict(self, tasks, **kwargs):
        logger.critical("start predictions")
        logger.critical(self.hostname)

        if not self.label_map :
            self.get_project_labels(tasks[0]['id'])

        if not self.is_custom:
            logger.critical(f"Exited because no custom model has been trained yet")
            return None, None
        
        # logger.critical(f"the model uses: {self.weights} to predict")
        results = []
        all_scores= []
        logger.critical(f"LABELS IN CONFIG: {self.labels}")
        for task in tasks:
            # logger.critical(task)
            
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url, project_dir=self.image_dir)
            img = Image.open(image_path)
            img_width, img_height = get_image_size(image_path)
            

            full_output = self.model.predict(img, imgsz=512,  device=DEVICE, conf = 0.3, iou = 0.5, show = False, max_det = 6) 
            xyxyns = full_output[0].boxes.xyxyn.cpu().numpy()
            confs = full_output[0].boxes.conf.cpu().numpy()
            classes = full_output[0].boxes.cls.cpu().numpy().astype(int)
 
 
            for i, xyxyn in enumerate(xyxyns):
                
                x_min, y_min, x_max, y_max, confidence, class_, label = float(xyxyn[0]), float(xyxyn[1]), float(xyxyn[2]), float(xyxyn[3]),  float(confs[i]), int(classes[i]), list(self.labels)[classes[i]]

                #add label name from label_map
                output_label = self.label_map.get(label, label)
                logger.critical("--"*20)
                logger.critical(f"Output Label {output_label}")
                if output_label not in self.labels:
                    logger.critical(output_label + ' label not found in project config.')
                    continue
                
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    "original_width": img.width,
                    "original_height": img.height,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [output_label],
                        'x': x_min * 100,
                        'y': y_min * 100,
                        'width': (x_max - x_min) * 100,
                        'height': (y_max - y_min) * 100
                    },
                    'score': confidence
                })
                all_scores.append(confidence)
                logger.critical(results)

        avg_score = sum(all_scores) / max(len(all_scores), 1)
        
        return results, avg_score

if __name__ == '__main__':
    '''
        执行测试，通过手动输入参数，测试模型
    '''
    kwargs = dict()

    model = yoloV8backend(**kwargs)

    model.fit(event=None,data=None, project=1)
    results = model.predict(tasks=[{
      "id": 2,
      "data": {
        "image": "/data/upload/1/e57d8b31-000002.jpg"
      },
      "annotations": [],
      "predictions": []
    }])
    print(results)
