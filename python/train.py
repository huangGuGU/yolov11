from ultralytics import YOLO
import torch
print(torch.cuda.is_available())
if __name__ == '__main__':
    # Load a model
    model = YOLO("../runs/detect/train34/weights/best.pt")
    
    # path = model.export(format="onnx")

    train_results = model.train(
        data="../dataset/data.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

    # # Evaluate model performance on the validation set
    # metrics = model.val()

    # # Perform object detection on an image
    # results = model("path/to/image.jpg")
    # results[0].show()

    # # Export the model to ONNX format
    # path = model.export(format="onnx")  # return path to exported model