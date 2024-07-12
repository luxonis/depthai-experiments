#include <array>
#include <vector>

class Point{
    //Used within the BoundingBox class when dealing with points.
    public:
    Point(float x,float y) : x(x),y(y){}    
    //Denormalize the point to pixel coordinates (0..frame width, 0..frame height)
    std::array<int,2> denormalize(std::vector<int> frame_shape){
        return {x * frame_shape[1], y * frame_shape[0]};
    }

    private:
    float x,y;
};


class BoundingBox{
    //This class helps with bounding box calculations. It can be used to calculate relative bounding boxes,
    //map points from relative to absolute coordinates and vice versa, crop frames, etc.
    public:
    BoundingBox(dai::ImgDetection bbox){
        xmin = bbox.xmin,ymin = bbox.ymin,xmax = bbox.xmax,ymax = bbox.ymax;
        width = xmax-xmin,height=ymax-ymin;
    }


    std::array<int,4> denormalize(std::vector<int> frame_shape){
        /*
        Denormalize the bounding box to pixel coordinates (0..frame width, 0..frame height).
        Useful when you want to draw the bounding box on the frame.

        Args:
            frame_shape: Shape of the frame (height, width)

        Returns:
            Tuple of two points (top-left, bottom-right) in pixel coordinates
        */
        return {
            (int)(frame_shape[1] * xmin),(int)(frame_shape[0] * ymin),
            (int)(frame_shape[1] * xmax),(int)(frame_shape[0] * ymax)
        };
    }

    Point map_point(float x,float y){
        /*
        Useful when you have a point inside the bounding box, and you want to map it to the frame.
        Example: You run face detection, create BoundingBox from the result, and also run
        facial landmarks detection on the cropped frame of the face. The landmarks are relative
        to the face bounding box, but you want to draw them on the original frame.

        Args:
            x: x coordinate of the point inside the bounding box (0..1)
            y: y coordinate of the point inside the bounding box (0..1)
        Returns:
            Point in absolute coordinates (0..1)
        */
        float mapped_x = xmin + width * x, mapped_y = ymin + height * y;
        return Point(mapped_x, mapped_y);
    }
    private:
    float xmin,ymin,xmax,ymax,width,height;
};