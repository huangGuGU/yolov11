#include "detect.hpp"
#include"preprocess.hpp"


int main() {

//    img_detect("/Users/hzh/Desktop/imgs","/Users/hzh/Desktop/inferred_img");
//    video_detect("/Users/hzh/Desktop/cat_video", "/Users/hzh/Desktop/inferred");
//    camera_detect();
//    spilt_video("/Users/hzh/Desktop/cat_video", "/Users/hzh/Desktop/label"
//                                                "","/Users/hzh/Desktop/img",5);
    delete_empty_label_file("/Users/hzh/Desktop/label",
                            "/Users/hzh/Desktop/img",
                            "/Users/hzh/Desktop/deletelabel",
                            "/Users/hzh/Desktop/newimg");
}









