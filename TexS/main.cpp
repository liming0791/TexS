#include <time.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

const double PI = 3.1415926;
const double E = 2.718281828459;

double AMat[10000];

class TN {
public:
	bool ifNode = false;
	Mat vec;
	TN* children[4];
	TN();
};

TN::TN() {
	for (int i = 0; i < 4; i++)
		this->children[i] = NULL;
}

void calcGMat(int w) { 
	float b = w / 2;

	double sigma = w / 12.f;
	float r2;
	float total = 0;
	for (int x = 0; x < w; x++) 
		for (int y = 0; y < w; y++)
		{
			r2 = (x - b)*(x - b) + (y - b) * (y - b);
			AMat[y * w + x] = 1 / sqrt(2*PI*sigma*sigma) * pow(E,-r2/2/sigma/sigma );

			printf("%f ", AMat[y * w + x]);
			if (y == w - 1) printf("\n");
			total += AMat[y * w + x];
		}

	printf("\ntotal: %f" , total);
}

void calcSMat(int w) {
	int b = w / 2;
	float r = 2 * b*b;
	float d;
	for (int x = 0; x < w; x++)
		for (int y = 0; y < w; y++)
		{
			d = (x - b)*(x - b) + (y - b)*(y - b);

			AMat[y * w + x] = r-d;
		}
}

void calcAMat(int w) {
	for (int x = 0; x < w; x++)
		for (int y = 0; y < w; y++)
		{
			AMat[y * w + x] = 1;
		}
}

double dis(Mat a, Mat b){
	int w = a.cols, 
		h = a.rows;
	double dis = 0;
	int a1, a2, a3, b1, b2, b3;
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			a1 = a.at<Vec3b>(x, y)[0];
			a2 = a.at<Vec3b>(x, y)[1];
			a3 = a.at<Vec3b>(x, y)[2];

			b1 = b.at<Vec3b>(x, y)[0];
			b2 = b.at<Vec3b>(x, y)[1];
			b3 = b.at<Vec3b>(x, y)[2];

			dis = dis + (a1 - b1)*(a1 - b1)
				+ (a2 - b2)*(a2 - b2)
				+ (a3 - b3)*(a3 - b3);
		}

	dis = sqrt(dis);

	return dis / h / w / 3 / 255;
}

int findMin(Mat& a, Mat s[4]) {
	float min = 1, d;
	int idx;

	for (int i = 0; i < 4; i++){
		d = dis(a, s[i]);
		if (d < min){
			min = d;
			idx = i;
		}
	}

	return idx;
}

Mat findCenter(vector<int>& idxs, vector<Mat>& windows){
	int w = windows[0].cols,
		h = windows[0].rows,
		size = idxs.size();
	Mat centerF(w, h, CV_32FC3, Scalar(0,0,0));
	Mat centerI(w, h, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 0; i < size; i++)
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
			{
				centerF.at<Vec3f>(x, y)[0] += windows[idxs[i]].at<Vec3b>(x, y)[0];
				centerF.at<Vec3f>(x, y)[1] += windows[idxs[i]].at<Vec3b>(x, y)[1];
				centerF.at<Vec3f>(x, y)[2] += windows[idxs[i]].at<Vec3b>(x, y)[2];
			}

	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			centerI.at<Vec3b>(x, y)[0] = centerF.at<Vec3f>(x, y)[0] / size;
			centerI.at<Vec3b>(x, y)[1] = centerF.at<Vec3f>(x, y)[1] / size;
			centerI.at<Vec3b>(x, y)[2] = centerF.at<Vec3f>(x, y)[2] / size;
		}

	return centerI;
}

void cluster(TN* t, vector<int>& w_i, vector<Mat>& windows){
	if (w_i.size() == 1){
		t->ifNode = true;
		return;
	}//endif
	else
	if (w_i.size() < 4){
		for (int i = 0; i < w_i.size(); i++)
		{
			t->children[i] = new TN();
			t->children[i]->vec = windows[w_i[i]];
			t->children[i]->ifNode = true;
		}
		return;
	}//endif

	vector<int> sub[4];
	Mat s[4];
	Mat c[4];
	bool done = false;
	double d;
	unsigned int count = 0;
	if (w_i.size() >= 4){
		
		for (int i = 0; i < 4; i++)
		{
			c[i] = windows[w_i[i]].clone();
		}

		do{
			count++;
			for (int i = 0; i < 4; i++)
			{
				s[i] = c[i].clone();
				sub[i] = vector<int>();
			}
			for (int i = 0, i_e = w_i.size(); i < i_e; i++)
			{
				int idx = findMin(windows[w_i[i]], s);
				sub[idx].push_back(w_i[i]);
			}
			for (int i = 0; i < 4; i++){
				c[i] = findCenter(sub[i], windows).clone();
			}
			done = true;
			for (int i = 0; i < 4 && done; i++){
			  d = dis(s[i], c[i]);
				if ( d > 0.00001) done = false;
			}
		} while (!done);

		if(count > 20)
		printf("cluster: %d times \n", count);

		//recursive action
		for (int i = 0; i < 4; i++)
		{
			t->children[i] = new TN();
			t->children[i]->vec = c[i].clone();
			cluster(t->children[i], sub[i], windows);
		}
	}//endif
};

void synthesis(Mat& l_image, Mat& c_image, TN* t, int b_size){
	TN* n_t = t, *d_t;
	float minDis = 10,d;
	int w = l_image.cols;
	double g,weight;

	Mat f_image(w,w, CV_32FC3, Scalar(0,0,0));

	for (int x = 0; x < w; x++)
		for (int y = 0; y < w; y++) 
		{
			c_image.at<float>(x, y) = 0;
			f_image.at<Vec3f>(x, y)[0] = 0;
			f_image.at<Vec3f>(x, y)[1] = 0;
			f_image.at<Vec3f>(x, y)[2] = 0;
		}
		

	for (int y = b_size; y < w - b_size ; y+= b_size/2)
		for (int x = b_size; x < w - b_size ; x+= b_size/2)
		{
			n_t = t;
			do{
				minDis = 10;
				for (int i = 0; i < 4; i++){
					if (n_t->children[i] != NULL){
						d = dis(Mat(l_image, 
								Range(x - b_size, x + b_size+1),
								Range(y-b_size,y+b_size+1)),
								n_t->children[i]->vec);
						if (d < minDis){
							minDis = d; d_t = n_t->children[i];
						}
					}
				}
				n_t = d_t;
			} while (!n_t->ifNode);

			//weight =  0.01/pow(minDis , 1.2);
			//printf("weight: %f \n", weight);
			//average 
			for (int yy = y - b_size; yy < y + b_size + 1; yy++)
				for (int xx = x - b_size; xx < x + b_size + 1; xx++)
				{
					//different weight
					weight = AMat[(yy - y + b_size) *(b_size * 2  + 1) + xx - x + b_size] *
									 0.01/pow(minDis , 1.2);
					
					//g = 1;
					f_image.at<Vec3f>(xx, yy)[0] += 
						weight*n_t->vec.at<Vec3b>(b_size + xx - x, b_size + yy -y)[0];
					f_image.at<Vec3f>(xx, yy)[1] += 
						weight*n_t->vec.at<Vec3b>(b_size + xx - x, b_size + yy - y)[1];
					f_image.at<Vec3f>(xx, yy)[2] += 
						weight*n_t->vec.at<Vec3b>(b_size + xx - x, b_size + yy - y)[2];

					c_image.at<float>(xx, yy) += weight;
				}

			//ji shu
			/*for (int yy = y - b_size; yy < y + b_size; yy++)
				for (int xx = x - b_size; xx < x + b_size; xx++)
				{
					f_image.at<Vec3f>(xx, yy)[0] +=
						n_t->vec.at<Vec3b>(b_size + xx - x, b_size + yy - y)[0];
					f_image.at<Vec3f>(xx, yy)[0] /= 2;

					f_image.at<Vec3f>(xx, yy)[1] +=
						n_t->vec.at<Vec3b>(b_size + xx - x, b_size + yy - y)[1];
					f_image.at<Vec3f>(xx, yy)[1] /= 2;

					f_image.at<Vec3f>(xx, yy)[2] +=
						n_t->vec.at<Vec3b>(b_size + xx - x, b_size + yy - y)[2];
					f_image.at<Vec3f>(xx, yy)[2] /= 2;
				}*/

		}

	float count;
	int val;

	//average
	for (int y = b_size; y < w - b_size; y++)
		for (int x = b_size; x < w - b_size; x++)
		{
			count= c_image.at<float>(x, y);
			val = f_image.at<Vec3f>(x, y)[0] / count;
			l_image.at<Vec3b>(x, y)[0] = val > 255 ? 255 : val;
			val = f_image.at<Vec3f>(x, y)[1] / count;
			l_image.at<Vec3b>(x, y)[1] = val > 255 ? 255 : val;
			val = f_image.at<Vec3f>(x, y)[2] / count;
			l_image.at<Vec3b>(x, y)[2] = val > 255 ? 255 : val;

			//printf("count: %f \n", count);
		}

	//ji shu
	/*for (int y = b_size; y < w - b_size; y++)
		for (int x = b_size; x < w - b_size; x++)
		{
			count = c_image.at<uchar>(x, y);
			val = f_image.at<Vec3f>(x, y)[0] ;
			l_image.at<Vec3b>(x, y)[0] = val > 255 ? 255 : val;
			val = f_image.at<Vec3f>(x, y)[1] ;
			l_image.at<Vec3b>(x, y)[1] = val > 255 ? 255 : val;
			val = f_image.at<Vec3f>(x, y)[2] ;
			l_image.at<Vec3b>(x, y)[2] = val > 255 ? 255 : val;
		}*/
}

int random(int a, int b){
	return (rand() % (b - a)) + a;
}

string getTimeStr()
{
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, 80, "%d-%m-%Y_%I-%M-%S", timeinfo);
	std::string str(buffer);

	return str;
}

int main(int argc, char *argv[])
{
	//parse params
	string s_image_name ;
	int w_size , f_w_size;
	int l_image_size , f_l_image_size;
	int i_times;

	if (argc < 2){
		cout << "Please enter a iamge name as a param" << endl;
		s_image_name = "leaf.jpg";
		f_l_image_size = 256;
	}
	else{
		s_image_name = argv[1];
		f_l_image_size = atoi(argv[2]);
	}

	//read image
	Mat r_image = imread(s_image_name);
	Mat s_image;
	cvtColor(r_image, s_image, CV_BGRA2BGR);
	//get image params
	int s_image_w = s_image.cols;
	int s_image_h = s_image.rows;

	int short_l = s_image_h < s_image_w ? s_image_h : s_image_w;

	f_w_size = short_l * 0.4;
	f_w_size = f_w_size % 2 == 0 ? (f_w_size + 1) : f_w_size;

	int b_size = f_w_size;

	Mat l_image;
	Mat t_image;

	string folderName = getTimeStr() + "_" + s_image_name + "_size_" + to_string(f_l_image_size);
	string folderCreateCommand = "mkdir " + folderName;
	std::system(folderCreateCommand.c_str());

	int l_times = 3;
	//multilevels
	for (int level = 0; level < l_times; level++) {

		switch (level)
		{
		case 0: b_size = 24;break;
		case 1: b_size = 16;break;
		case 2: b_size = 12;break;
		default:
			break;
		}
		w_size = 2 * b_size + 1;

		l_image_size = (f_l_image_size >> (l_times - 1 - level)) + 2 * b_size;

		//init large image
		if (level == 0)
		{
			int sx, sy, xxx, yyy;
			l_image = Mat(l_image_size, l_image_size, CV_8UC3, Scalar(255, 255, 255));
			for (int x = 0; x < l_image_size; x += b_size << 1)
				for (int y = 0; y < l_image_size; y += b_size << 1)
				{
					sx = random(0, s_image_w - 2 * b_size - 1);
					sy = random(0, s_image_h - 2 * b_size - 1);

					for (int xx = x; xx < x + 2 * b_size; xx++)
						for (int yy = y; yy < y + 2 * b_size; yy++)
						{
							if (xx < l_image_size && yy < l_image_size)
							{
								xxx = sx + xx - x;
								yyy = sy + yy - y;
								assert(xxx >= 0 && xxx < s_image_w &&
									yyy >= 0 && yyy < s_image_h && " small image index overflow ! ");
								if (xx < 0 || yy < 0 ||
									xxx < 0 || yyy < 0 ||
									xxx >= s_image_w || yyy >= s_image_h)
								{
									int i = 0;
								}

								l_image.at<Vec3b>(yy, xx) = s_image.at<Vec3b>(yyy, xxx);
							}
						}
				}
		}
		else
		{
			int l_b_size;
			switch (level)
			{
			case 1: l_b_size = 24;break;
			case 2: l_b_size = 16;break;
			default:
				break;
			}
			int l_l_image_size = (f_l_image_size >> (l_times - level)) + 2 * l_b_size;
			t_image = Mat(l_image, Range(l_b_size, l_l_image_size - l_b_size), Range(l_b_size, l_l_image_size - l_b_size));
			l_image = Mat(l_image_size, l_image_size, CV_8UC3, Scalar(255, 255, 255));
			resize(t_image,
						 l_image,
						 Size(l_image_size,l_image_size));
		}
		//init counter
		Mat count_image(l_image_size, l_image_size, CV_32FC1);
		for (int x = 0; x < l_image_size; x++)
			for (int y = 0; y < l_image_size; y++)
			count_image.at<float>(x, y) = 0;

		//get windows
		vector<Mat> windows = vector<Mat>();
		for (int x = b_size, x_e = s_image_w - b_size; x < x_e; x++)
			for (int y = b_size, y_e = s_image_h - b_size; y < y_e; y++)
			windows.push_back(Mat(s_image, Range(y - b_size, y + b_size + 1), Range(x - b_size, x + b_size + 1)));
		
		//build tree 
		TN * Tree = new TN();
		vector<int> idxs;
		for (int i = 0; i < windows.size(); i++)
			idxs.push_back(i);
		cluster(Tree, idxs, windows);

		//calc GMat
		calcGMat(w_size);
		//return 0;

		//calcAMat(w_size);
		//calcSMat(w_size);
		//init
		stringstream ss;
		string name;
		string fullPath;
		//while (waitKey(-1) != 27)
		int i_times = 50;
		for (int i = 0; i < i_times; i++)
		{
			synthesis(l_image, count_image, Tree, b_size);
		}
		name = "_wsize_" + to_string(w_size) + "_iterate_"+ to_string(i_times) + ".jpg";
		ss << folderName << "/" << name;
		fullPath = ss.str();
		ss.str("");
		imwrite(fullPath, l_image);
	}

	//show image
	namedWindow("small image");
	imshow("small image", s_image);
	namedWindow("large image");
	imshow("large image", l_image);
	while (waitKey(-1) != 27);
	return 0;
}