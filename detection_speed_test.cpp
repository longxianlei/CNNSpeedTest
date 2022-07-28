// Author: Xianlei Long, 2022-07-14.
// There include processing speed tests on different object detection models.
// Include YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv4-tiny, YOLOv7-tiny, FasterRCNN_RestNet50,FasterRCNN_Inception.
// Just load different model weights and their corresponding config file. Then, statistic the detection time.
// Note, to reduce the processing cost, you should load the model early, and use the model to detect once.
// (warmming up is improtant).

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <opencv2/imgproc.hpp>


using namespace std;
using namespace cv;

float conf_threshold = 0.5; // the confidence threshold.
float nms_threshold = 0.4; // NMS threshold.
int input_width = 416; // input image width.
int input_height = 416; // input image height.
vector<string> classes; // the coco class names.
bool is_show_img = false; // whether show the image.
int runnning_times = 50; // the detection running loop times for the detector.
bool is_use_gpu = false; // use GPU or cpu.

std::vector<String> GetOutputsNames(const cv::dnn::Net& net);
void PostProcess(Mat& frame, const vector<Mat>& outs);
void DrawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
void TestFasterRCNN(string img_path, string model_name);
int TestYoloSeries(string img_path, string model_name);

int main()
{
	// Load names of classes
	string classesFile = "C:\\CODEREPO\\DahuaGal\\model\\coco.names";
	std::ifstream ifs(classesFile.c_str());
	string line;
	while (std::getline(ifs, line)) classes.push_back(line);
	// Test detection model.
	string image_path = "./image/1111all_new32.jpg"; //"test_img3.bmp";  // test.jpg  car.jpg  1111all_new32.jpg car2.bmp
	// Load the CNN model and its' config file. (YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv4-tiny, YOLOv7-tiny series.)
	string yolo_model_name = "yolov4-tiny";
	// ID 0: inception1, ID 1: faster_rcnn_inception Faster RCNN-Inception, ID 2: faster_rcnn_resnet50 Faster RCNN-ResNet50;  
	string rcnn_model_name = "Faster RCNN-Inception";
	TestYoloSeries(image_path, yolo_model_name); // test yolo model speed.
	//TestFasterRCNN(image_path, rcnn_model_name); // test faster-rcnn speed.
}

int TestYoloSeries(string img_path, string model_name)
{
	//// Load the CNN model and its' config file. (YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv4-tiny, YOLOv7-tiny series.)
	//string yolo_model_name = "yolov4";
	cv::String cfg_file = "C:\\CODEREPO\\DahuaGal\\model\\" + model_name + ".cfg";
	cv::String weights_file = "C:\\CODEREPO\\DahuaGal\\model\\" + model_name + ".weights";
	cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfg_file, weights_file);

	// Use GPU.
	if (is_use_gpu)
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	else
	{
		// Use CPU.
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	cv::Mat grab_img = cv::imread(img_path);
	cout << "image height: " << grab_img.rows << ", width:" << grab_img.cols << endl;

	// Warm up, load the model and run once.
	cv::Mat blob1 = cv::dnn::blobFromImage(grab_img, 1.0 / 255.0, { 224, 224 }, 0.00392, true);
	net.setInput(blob1);
	vector<Mat> detectionMat1;
	try {
		net.forward(detectionMat1, GetOutputsNames(net));
	}
	catch (cv::Exception& e) {
		return false;
	}

	// Begin the timer.
	chrono::steady_clock::time_point begin_detect = chrono::steady_clock::now();
	for (int i = 0; i < runnning_times; i++)
	{
		cv::Mat blob = cv::dnn::blobFromImage(grab_img, 1.0 / 255.0, { 224,224 }, 0.00392, true);
		net.setInput(blob);
		vector<Mat> detectionMat;
		try {
			//net.forward(detectionMat, GetOutputsNames(net));
			net.forward(detectionMat, net.getUnconnectedOutLayersNames());
		}
		catch (cv::Exception& e) {
			return false;
		}
		if (is_show_img)
		{
			PostProcess(grab_img, detectionMat);
			cv::imwrite("detect_" + model_name + ".png", grab_img);
			cv::imshow(model_name + " result show", grab_img);
			cv::waitKey(0);
		}

	}
	// End the timer, and cout the processing speed and fps.
	chrono::steady_clock::time_point end_detect = chrono::steady_clock::now();
	cout << "Model: " << model_name << endl;
	cout << "The detect process is  get clock (ms): !!!!!!!!!!!!!" << chrono::duration_cast<chrono::microseconds>(end_detect - begin_detect).count() / 1000.0 << endl;
	cout << "Processing speed: " << runnning_times * 1000000.0 / float(chrono::duration_cast<chrono::microseconds>(end_detect - begin_detect).count()) << " FPS!" << endl;
}

vector<cv::String> GetOutputsNames(const cv::dnn::Net& net)
{
	static vector<string> names;
	if (names.empty()) {
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<string> layersNames = net.getLayerNames();

		//Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void PostProcess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > conf_threshold)
			{
				//cout << "frame.cols" << frame.cols << ", rows:" << frame.rows << endl;
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}
	// Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
	vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		DrawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
	}
}

// Draw the predicted bounding box
void DrawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}
	//cout << classId << ", " << label << endl;
	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}


class faster_rcnn
{
public:
	faster_rcnn(float conf_threshold, string modelID)
	{
		this->conf_threshold = conf_threshold;
		if (modelID == "Faster RCNN")
		{
			this->net = cv::dnn::readNet("faster_rcnn.pb", "faster_rcnn.pbtxt"); // detect ID card.
		}
		else if (modelID == "Faster RCNN-Inception")
		{
			this->net = cv::dnn::readNetFromTensorflow("./weights/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb",
				"./weights/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"); // trained on COCO.
		}
		else
		{
			this->net = cv::dnn::readNetFromTensorflow("./weights/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb",
				"./weights/faster_rcnn_resnet50_coco_2018_01_28.pbtxt"); //trained on COCO.
		}

		if (is_use_gpu)
		{
			this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
			this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
		}
		else
		{
			// Use CPU.
			this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
			this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		}
	}
	void detect(Mat& frame);
private:
	float conf_threshold;
	// ID 0: inception1, ID 1: faster_rcnn_inception, ID 2: faster_rcnn_resnet50; 
	string modelID;
	cv::dnn::Net net;
};

void faster_rcnn::detect(Mat& frame)
{
	Mat blob = cv::dnn::blobFromImage(frame);
	//Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, { 224,224 });
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> classes_id;
	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* pdata = (float*)outs[i].data;
		int num_proposal = outs[i].size[2];
		int len = outs[i].size[3];
		for (int n = 0; n < num_proposal; n++)
		{
			const int objectClass = int(pdata[1]);
			const float score = pdata[2];
			if (score > this->conf_threshold)
			{
				const int left = int(pdata[3] * frame.cols);
				const int top = int(pdata[4] * frame.rows);
				const int right = int(pdata[5] * frame.cols);
				const int bottom = int(pdata[6] * frame.rows);
				classes_id.push_back(objectClass);
				confidences.push_back(score);
				boxes.push_back(Rect(left, top, right - left, bottom - top));
			}
			pdata += len;
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
	vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
	//cout << "detected boxes: " << boxes.size() << ", filtered boxes: " << indices.size() << endl;

	if (is_show_img)
	{
		//for (int i = 0; i < boxes.size(); i++)
		for (size_t i = 0; i < indices.size(); ++i)
		{
			int idx = indices[i];
			//Rect box = boxes[idx];
			//Draw a rectangle displaying the bounding box
			rectangle(frame, Point(boxes[idx].x, boxes[idx].y), Point(boxes[idx].x + boxes[idx].width, boxes[idx].y + boxes[idx].height), Scalar(255, 0, 255), 2);
			//Get the label for the class name and its confidence
			string label = format("%.2f", confidences[idx]);
			label = classes[classes_id[idx]] + ":" + label;
			//cout << "calss id: " << classes_id[i] << endl;
			//Display the label at the top of the bounding box
			int baseLine;
			Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			int top = max(boxes[idx].y, labelSize.height);
			rectangle(frame, Point(boxes[idx].x, top - int(1.5 * labelSize.height)), Point(boxes[idx].x + int(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
			putText(frame, label, Point(boxes[idx].x, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
		}
	}
}

void TestFasterRCNN(string img_path, string model_name)
{
	//// ID 0: inception1, ID 1: faster_rcnn_inception Faster RCNN-Inception, ID 2: faster_rcnn_resnet50 Faster RCNN-ResNet50;  
	//string model_name = "Faster RCNN-ResNet50";
	faster_rcnn net(0.6, model_name);

	// Warm up the detector.
	//string imgpath = "./image/test_img3.bmp";  // test.jpg  car.jpg  1111all_new32.jpg
	Mat srcimg_temp = imread(img_path);

	net.detect(srcimg_temp);

	// Begin detect, count the timer.
	Mat srcimg = imread(img_path);
	//resize(srcimg, srcimg, Size(1024, 1024), 0, 0, 1);
	chrono::steady_clock::time_point begin_detect = chrono::steady_clock::now();
	for (int i = 0; i < runnning_times; i++)
	{
		net.detect(srcimg);
	}
	chrono::steady_clock::time_point end_detect = chrono::steady_clock::now();
	cout << "Model: " << model_name << endl;
	cout << "The detect process is  get clock (ms): !!!!!!!!!!!!!" << chrono::duration_cast<chrono::microseconds>(end_detect - begin_detect).count() / 1000 << endl;
	cout << "Processing speed: " << runnning_times * 1000000.0 / float(chrono::duration_cast<chrono::microseconds>(end_detect - begin_detect).count()) << " FPS!" << endl;

	if (is_show_img)
	{
		imwrite("detect_" + model_name + ".png", srcimg);
		cv::imshow(model_name + " result show", srcimg);
		waitKey(0);
		destroyAllWindows();
	}
}
