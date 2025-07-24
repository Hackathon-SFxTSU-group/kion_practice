import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2


class FaceReId:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Детектор лиц (MTCNN) и эмбеддер (ResNet от FaceNet)
        self.detector = MTCNN(image_size=160, margin=20, device=self.device)
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def analyze_persons(self, track_faces):
        track_embeddings = {}

        for track_id, crops in tqdm(track_faces.items(), desc="Extracting face embeddings"):
            embeddings = []

            for face_img in crops:
                img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                aligned_face = self.detector(img_pil)
                if aligned_face is not None:
                    aligned_face = aligned_face.to(self.device)
                    with torch.no_grad():
                        emb = self.embedder(aligned_face.unsqueeze(0))
                    embeddings.append(emb.squeeze().cpu().numpy())

            if embeddings:
                track_embeddings[track_id] = np.mean(embeddings, axis=0)

        return self.cluster_track_ids(track_embeddings)

    def cluster_track_ids(self, track_embeddings):
        X = list(track_embeddings.values())
        ids = list(track_embeddings.keys())

        clustering = DBSCAN(eps=0.7, min_samples=1, metric='cosine').fit(X)
        return {track_id: f"person_{label}" for track_id, label in zip(ids, clustering.labels_)}
