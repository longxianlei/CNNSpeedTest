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
bool is_show_img = true; // whether show the image.
int runnning_times = 100; // the detection running loop times for detector.

std::vector<cv::String> GetOutputsNames(const cv::dnn::Net& net);
void PostProcess(Mat& frame, const vector<Mat>& outs);
void DrawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
void TestFasterRCNN(); // test faster-rcnn speed.
int TestYoloSeries(); // test yolo speed.

/*/
// Xianlei Long, 2022-07-14.
// There include processing speed tests on different object detection models.
// Include YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv4-tiny, FasterRCNN_RestNet50,FasterRCNN_Inception.
// Just load different model weights and their corresponding config file. Then, statistic the detection time.
// Note, to reduce the processing cost, you should load the model early, and use the model to detect once.
// (warmming up is improtant).
/*/

int main()
{
	// Load names of classes
	string classesFile = "C:\\CODEREPO\\DahuaGal\\model\\coco.names";
	std::ifstream ifs(classesFile.c_str());
	string line;
	while (std::getline(ifs, line)) classes.push_back(line);
	// Test detection model speed.
	is_show_img = false;
	TestYoloSeries();
	TestFasterRCNN();
}

int TestYoloSeries()
{
	// Load the CNN model and its' config file. (YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv4-tiny series.) You can use other model and config file.
	cv::String cfg_file = "C:\\CODEREPO\\DahuaGal\\model\\yolov3-tiny.cfg";
	cv::String weights_file = "C:\\CODEREPO\\DahuaGal\\model\\yolov3-tiny.weights";
	cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfg_file, weights_file);

	// Use GPU.
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

	// Use CPU.
	//net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	//net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	cv::Mat grab_img = cv::imread("test_img3.bmp");  //"test_img3.bmp";  // test.jpg  car.jpg  1111all_new32.jpg
	cout << "image height: " << grab_img.rows << ", width:" << grab_img.cols << endl;

	// Warm up, load the model and run once.
	cv::Mat blob1 = cv::dnn::blobFromImage(grab_img, 1.0 / 255.0, { 224,224 }, 0.00392, true);
	net.setInput(blob1);
	vector<Mat> detectionMat1;
	try {
		net.forward(detectionMat1, GetOutputsNames(net));
	}
	catch (cv::Exception& e) {
		return false;
	}

	// Begin the timer, begin the speed test.
	chrono::steady_clock::time_point begin_detect = chrono::steady_clock::now();
	for (int i = 0; i < runnning_times; i++)
	{
		cv::Mat blob = cv::dnn::blobFromImage(grab_img, 1.0 / 255.0, { 224,224 }, 0.00392, true);
		net.setInput(blob);
		vector<Mat> detectionMat;
		try {
			net.forward(detectionMat, GetOutputsNames(net));
		}
		catch (cv::Exception& e) {
			return false;
		}
		if (is_show_img)
		{
			PostProcess(grab_img, detectionMat);
			cv::imshow("show", grab_img);
			cv::waitKey(0);
		}

	}
	// End the timer, and cout the processing speed and fps.
	chrono::steady_clock::time_point end_detect = chrono::steady_clock::now();
	cout << "The detect process is  get clock (ms): !!!!!!!!!!!!!" << chrono::duration_cast<chrono::microseconds>(end_detect - begin_detect).count() / 1000 << endl;
	cout << "Processing speed: " << runnning_times*1000000.0 / float(chrono::duration_cast<chrono::microseconds>(end_detect - begin_detect).count()) << " FPS!" << endl;
}

vector<cv::String> GetOutputsNames(const cv::dnn::Net& net)
{
	static vector<String> names;
	if (names.empty()) {
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

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
				cout << "frame.cols" << frame.cols << ", rows:" << frame.rows << endl;
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
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}
	cout << classId << ", " << label << endl;
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
	faster_rcnn(float conf_threshold, int modelID)
	{
		this->conf_threshold = conf_threshold;
		if (modelID == 0)
		{
			this->net = cv::dnn::readNet("faster_rcnn.pb", "faster_rcnn.pbtxt"); // detect ID card.
		}
		else if (modelID == 1)
		{
			this->net = cv::dnn::readNetFromTensorflow("./weights/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb",
				"./weights/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"); // trained on COCO.
		}
		else
		{
			this->net = cv::dnn::readNetFromTensorflow("./weights/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb",
				"./weights/faster_rcnn_resnet50_coco_2018_01_28.pbtxt"); //trained on COCO.
		}
		this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

	}
	void detect(Mat& frame);
private:
	float conf_threshold;
	// ID 0: inception1, ID 1: faster_rcnn_inception, ID 2: faster_rcnn_resnet50; 
	int modelID;
	cv::dnn::Net net;
};

void faster_rcnn::detect(Mat& frame)
{
	Mat blob = cv::dnn::blobFromImage(frame);
	//Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, { 256,256}, 0.00392, true);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	// post processing.
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

	if (is_show_img)
	{
		for (int i = 0; i < boxes.size(); i++)
		{
			//Draw a rectangle displaying the bounding box
			rectangle(frame, Point(boxes[i].x, boxes[i].y), Point(boxes[i].x + boxes[i].width, boxes[i].y + boxes[i].height), Scalar(0, 0, 255), 2);
			//Get the label for the class name and its confidence
			string label = format("%.2f", confidences[i]);
			label = classes[classes_id[i]] + label;
			//cout << "calss id: " << classes_id[i] << endl;
			//Display the label at the top of the bounding box
			int baseLine;
			Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			int top = max(boxes[i].y, labelSize.height);
			//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
			putText(frame, label, Point(boxes[i].x, top - 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);
		}
	}
}

void TestFasterRCNN()
{
	int modelID = 2; // ID 0: inception1, ID 1: faster_rcnn_inception, ID 2: faster_rcnn_resnet50; 
	faster_rcnn net(0.6, modelID);

	// Warm up the detector.
	string imgpath = "test_img3.bmp";  // test.jpg  car.jpg  1111all_new32.jpg
	Mat srcimg_temp = imread(imgpath);
	//resize(srcimg, srcimg, Size(256, 256));
	net.detect(srcimg_temp);

	// Begin detect, count the timer.
	Mat srcimg = imread(imgpath);
	chrono::steady_clock::time_point begin_detect = chrono::steady_clock::now();
	for (int i = 0; i < runnning_times; i++)
	{
		net.detect(srcimg);
	}
	chrono::steady_clock::time_point end_detect = chrono::steady_clock::now();
	cout << "The detect process is  get clock (ms): !!!!!!!!!!!!!" << chrono::duration_cast<chrono::microseconds>(end_detect - begin_detect).count() / 1000 << endl;
	cout << "Processing speed: " << runnning_times*1000000.0 / float(chrono::duration_cast<chrono::microseconds>(end_detect - begin_detect).count()) << " FPS!" << endl;
	cout << "end detect!!!" << endl;

	if (is_show_img)
	{
		imshow("test", srcimg);
		waitKey(3000);
		destroyAllWindows();
	}
}
