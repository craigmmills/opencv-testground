//
// The full "Square Detector" program.
// It loads several images subsequentally and tries to find squares in
// each image
//
#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#endif


int thresh = 50;
IplImage* img = 0;
IplImage* img0 = 0;
CvMemStorage* storage = 0;
CvPoint pt[4];

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2 
double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 )
{
    double dx1 = pt1->x - pt0->x;
    double dy1 = pt1->y - pt0->y;
    double dx2 = pt2->x - pt0->x;
    double dy2 = pt2->y - pt0->y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


//CM  function to draw out each of the stages
void draw_stuff(IplImage* draw_img)
{
    
    //set up rnadom window name to capture each stage of the process
    char my_letters[] = "abcdefghijklmnopqrstuvwxyz";
    char ran_str[15] = "image:";
    
    srand((unsigned)time(NULL)); //seed rand
    
    int i;
    int t;
    
    
    //temp string constant to convert single char to string
    char temp[2];
    temp[1] = '\0';
    
    
    //concatenate rand chars to string
    for(i = 0; i < 8;i++){
        
        t = rand() % 27;
        temp[0] = my_letters[t];

        /* or however long you want it */
        strcat(ran_str, temp );
    }
    
    // create window with the random generated name
    cvNamedWindow( ran_str, 1 );
    
    printf("%s\n",ran_str);
    //take copy of image
    IplImage* cpy = cvCloneImage( draw_img );
    
    // show the resultant image in new window
    cvShowImage(ran_str,cpy);
    cvReleaseImage( &cpy );
    cvWaitKey(0);
    cvDestroyWindow(ran_str);
}





//try a different technique using obvious clumps of colour in the colour histogram
void createHistogram(IplImage* src)
{
    
    // Compute the HSV image and decompose it into separate planes.
    //
    IplImage* hsv = cvCreateImage( cvGetSize(src), 8, 3 );
    cvCvtColor( src, hsv, CV_BGR2HSV );
    IplImage* h_plane = cvCreateImage( cvGetSize(src), 8, 1 );
    IplImage* s_plane = cvCreateImage( cvGetSize(src), 8, 1 );
    IplImage* v_plane = cvCreateImage( cvGetSize(src), 8, 1 );
    IplImage* planes[] = { h_plane, s_plane };
    
        
    cvCvtPixToPlane( hsv, h_plane, s_plane, v_plane, 0 );
    //draw_stuff(hsv);
    //draw_stuff(h_plane);
    draw_stuff(s_plane);
    //draw_stuff(v_plane);
    
    // Build the histogram and compute its contents.
    //
    int h_bins = 30, s_bins = 32;
    
    CvHistogram* hist;
    {
        int hist_size[] = { h_bins, s_bins };
        float h_ranges[] = { 0, 180 }; // hue is [0,180]
        float s_ranges[] = { 0, 255 };
        float* ranges[] = { h_ranges, s_ranges };
        hist = cvCreateHist(
                            2,
                            hist_size,
                            CV_HIST_ARRAY,
                            ranges,
                            1
                            );
    }
    
    
    cvCalcHist( planes, hist, 0, 0 ); //Compute histogram
    
    
    cvNormalizeHist( hist, 1.0 ); //Normalize it
    // Create an image to use to visualize our histogram.
    //
    int scale = 10;
    IplImage* hist_img = cvCreateImage(
                                       cvSize( h_bins * scale, s_bins * scale ),
                                       8,
                                       3
                                       );
    cvZero( hist_img );
    // populate our visualization with little gray squares.
    //
    float max_value = 0;
    cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 );
    
    for( int h = 0; h < h_bins; h++ ) {
        for( int s = 0; s < s_bins; s++ ) {
            float bin_val = cvQueryHistValue_2D( hist, h, s );
            int intensity = cvRound( bin_val * 255 / max_value );
            cvRectangle(
                        hist_img,
                        cvPoint( h*scale, s*scale ),
                        cvPoint( (h+1)*scale - 1, (s+1)*scale - 1),
                        CV_RGB(intensity,intensity,intensity),
                        CV_FILLED,8,0
                        );
        }
    }
    
    cvNamedWindow( "orig", 1 );
    
    cvShowImage( "orig", src );
    cvNamedWindow( "H-S Histogram", 1 );        
    cvShowImage( "H-S Histogram", hist_img );
    cvWaitKey(0);    
}


// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
CvSeq* findSquares4( IplImage* img, CvMemStorage* storage )
{
    
        
    CvSeq* contours;
    int i, c, l, N = 11;
    CvSize sz = cvSize( img->width & -2, img->height & -2 );
    IplImage* timg = cvCloneImage( img ); // make a copy of input image
    IplImage* gray = cvCreateImage( sz, 8, 1 ); 
    IplImage* pyr = cvCreateImage( cvSize(sz.width/2, sz.height/2), 8, 3 );
    IplImage* tgray;
    CvSeq* result;
    double s, t;
    // create empty sequence that will contain points -
    // 4 points per square (the square's vertices)
    CvSeq* squares = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );
    
    // select the maximum ROI in the image
    // with the width and height divisible by 2
    cvSetImageROI( timg, cvRect( 0, 0, sz.width, sz.height ));
    
    // down-scale and upscale the image to filter out the noise
    cvPyrDown( timg, pyr, 7 );
    cvPyrUp( pyr, timg, 7 );
    tgray = cvCreateImage( sz, 8, 1 );
    
    
    
    // find squares in every color plane of the image
    for( c = 0; c < 3; c++ )
    {
        // extract the c-th color plane
        cvSetImageCOI( timg, c+1 );
        
        cvCopy( timg, tgray, 0 );
        
                
        // try several threshold levels
        for( l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading   
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging) 
                cvCanny( tgray, gray, 100, 203, 5 );
                printf("%d\n",thresh);
                
                draw_stuff(gray);

                
                // dilate canny output to remove potential
                // holes between edge segments 
                cvDilate( gray, gray, 0, 1 );
                
                
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                cvThreshold( tgray, gray, (l+1)*255/N, 255, CV_THRESH_BINARY );
                
            }
            
            // find contours and store them all as a list
            cvFindContours( gray, storage, &contours, sizeof(CvContour),
						   CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
            
       
            
            // test each contour
            while( contours )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                result = cvApproxPoly( contours, sizeof(CvContour), storage,
									  CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );
                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
				
				
                if( result->total == 4 &&
				   fabs(cvContourArea(result,CV_WHOLE_SEQ,0)) > 1000 &&
				   cvCheckContourConvexity(result) )
                {
                    s = 0;
                    
                    for( i = 0; i < 5; i++ )
                    {
                        // find minimum angle between joint
                        // edges (maximum of cosine)
                        if( i >= 2 )
                        {
                            t = fabs(angle(
										   (CvPoint*)cvGetSeqElem( result, i),
										   (CvPoint*)cvGetSeqElem( result, i-2),
										   (CvPoint*)cvGetSeqElem( result, i-1)));
                            s = s > t ? s : t;
                        }
                    }
                    
                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence 
                    if( s < 0.3 )
                        for( i = 0; i < 4; i++ )
                            cvSeqPush( squares,
									  (CvPoint*)cvGetSeqElem( result, i));
                }
                
                // take the next contour
                contours = contours->h_next;
            }
        }
    }
    

    
    
    // release all the temporary images
    
      
    cvReleaseImage( &gray );
    cvReleaseImage( &pyr );
    cvReleaseImage( &tgray );
    cvReleaseImage( &timg );
    
    return squares;
}


// the function draws all the squares in the image
void drawSquares( IplImage* img, CvSeq* squares )
{
    CvSeqReader reader;
    IplImage* cpy = cvCloneImage( img );
    int i;
    
    // initialize reader of the sequence
    cvStartReadSeq( squares, &reader, 0 );
    
    // read 4 sequence elements at a time (all vertices of a square)
    for( i = 0; i < squares->total; i += 4 )
    {
        CvPoint* rect = pt;
        int count = 4;
        
        // read 4 vertices
        memcpy( pt, reader.ptr, squares->elem_size );
        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );
        memcpy( pt + 1, reader.ptr, squares->elem_size );
        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );
        memcpy( pt + 2, reader.ptr, squares->elem_size );
        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );
        memcpy( pt + 3, reader.ptr, squares->elem_size );
        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );
        
        // draw the square as a closed polyline 
		
        cvPolyLine( cpy, &rect, &count, 1, 1, CV_RGB(0,255,0), 3, 8, 0 );
    }
    
    // show the resultant image
    cvShowImage("image",cpy);
    cvReleaseImage( &cpy );
    
    
    
    
    
}




void on_trackbar( int a )
{
    if( img )
        drawSquares( img, findSquares4( img, storage ) );
}

char* names[] = { "/Users/craigmills/Documents/code/c/opencv_with_postits/opencv_with_postits/postits.jpg" };

int main(int argc, char** argv)
{
    int i;
    // create memory storage that will contain all the dynamic data
    storage = cvCreateMemStorage(0);
    // create window with name "image"
    cvNamedWindow( "image", 1 );
    // create trackbar (slider) with parent "image" and set callback
    // (the slider regulates upper threshold, passed to Canny edge detector) 
    //cvCreateTrackbar( "thresh1", "image", &thresh, 1000, on_trackbar );
    
    for( i = 0; i < 6; i++ )
    {
        // load i-th image
        img0 = cvLoadImage( names[i], 1 );
        if( !img0 )
        {
            printf("Couldn't load %s\n", names[i] );
            break;
        }
        img = cvCloneImage( img0 );
        
        //create histogram of saturation to take a look at the distribution- see if we can identify patterns
        
        //createHistogram(img);
        
        // force the image processing
        drawSquares( img, findSquares4( img, storage ) );
        //on_trackbar(0);
        // wait for key.
        // Also the function cvWaitKey takes care of event processing
        cvWaitKey(0);
        // release both images
        cvReleaseImage( &img );
        cvReleaseImage( &img0 );
        // clear memory storage - reset free space position
        cvClearMemStorage( storage );
    }
    
    cvDestroyWindow("image");
    
    return 0;
}

#ifdef _EiC
main(1,"squares.c");
#endif
