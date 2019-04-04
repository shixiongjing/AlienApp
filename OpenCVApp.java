/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package opencvapp;

import Jama.Matrix;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Image;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
/**
 *
 * @author Kyle
 */
public class OpenCVApp {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat mat = Mat.eye(3, 3, CvType.CV_8UC1);
        System.out.println("mat = " + mat.dump());   
            
        // Image Codecs Class For Img Processing
        Imgcodecs imageCodecs = new Imgcodecs();
       // Mat foreground = Imgcodecs.imread("ex_fg_4.jpg");
        Mat foreground = Imgcodecs.imread("nine_corners.png");
        Mat foreground_resized = new Mat();
        Size size_image = foreground.size();
        Size sz = new Size(510, 553);
        //Size sz = new Size(300,400);
        Imgproc.resize(foreground, foreground_resized, sz );

        // Change to Image
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", foreground_resized, matOfByte);
        byte[] byteArray = matOfByte.toArray();

        InputStream in = new ByteArrayInputStream(byteArray);
        BufferedImage bufImage = ImageIO.read(in);
        
        // Display Foreground Image
        JFrame frame = new JFrame(); 
        ImageIcon imageIcon = new ImageIcon(bufImage);
        Image image = imageIcon.getImage();
        frame.getContentPane().add(new JLabel(new ImageIcon(bufImage))); 
        frame.pack(); 
        //frame.setVisible(true);
        System.out.println("Image Loaded"); 
        
        // Convert to Grayscale 
         Mat gray = new Mat();

        //Converting the image to gray scale and saving it in the dst matrix
        Imgproc.cvtColor(foreground_resized, gray, Imgproc.COLOR_RGB2GRAY);
        
        // Canny Edge Detection
        Mat edges = new Mat();
        //Imgproc.Canny(gray, edges, 60, 60*3);
      // Imgproc.Canny(gray,edges,50,100);
        //Imgcodecs.imwrite("edges.jpg", edges);
        
       Imgproc.Canny(gray, edges, 50, 200, 3, false);
        Imgcodecs.imwrite("edges.jpg", edges);
        
        // Hough Transform
        Mat lines = new Mat();
        int rows = edges.rows();
        int cols = edges.cols();
        int max_r = (int) Math.sqrt(Math.pow(rows,2) + Math.pow(cols, 2));
        int max_theta = 180;
        System.out.println(max_r);
        System.out.println(max_theta);
        
        // Get Edge Points
        int[][] H = new int[max_r + 1][max_theta + 1];
        ArrayList<Integer> edge_point_rows = new ArrayList<Integer>();
        ArrayList<Integer> edge_point_cols = new ArrayList<Integer>();
        for(int r = 0; r < rows; r++){
            for(int c = 0; c < cols; c++){
              double [] d = edges.get(r, c);
              if(d[0] > 0.0){
                  edge_point_rows.add(r);
                  edge_point_cols.add(c);
              }
            }
        }
        
        // Accumulate
        int theta_offset = 5;
        for(int i = 0; i < edge_point_rows.size(); i++){
            int x = edge_point_cols.get(i);
            int y = edge_point_rows.get(i);
            
            for(int theta = 0; theta < 181;  theta = theta + theta_offset){
                double angle_rad = theta * Math.PI / 180.0;
                int r = Math.abs((int)(x * Math.cos(angle_rad) + y * Math.sin(angle_rad)));

                H[r][theta] = H[r][theta] + 1;
            }
        }
        
        // Voting
        // CHANGED THIS
        int accumulator_threshold = 40;
        ArrayList<Integer> r_vals = new ArrayList<Integer>();
        ArrayList<Integer> theta_vals = new ArrayList<Integer>();
        for(int i = 0; i < max_r; i++){
            for(int j = 0; j < max_theta + 1; j++){
                if(H[i][j] > accumulator_threshold){
                    r_vals.add(i);
                    theta_vals.add(j);
                }
            }
        }
        
        // Draw Lines
          Mat pooled = foreground_resized.clone();
          Mat corner_img = foreground_resized.clone();
          Mat img_pt = foreground_resized.clone();
        ArrayList<Integer> horz_line_r = new ArrayList<Integer>();
        ArrayList<Integer> vert_line_r = new ArrayList<Integer>();
        ArrayList<Integer> horz_line_theta = new ArrayList<Integer>();
        ArrayList<Integer> vert_line_theta = new ArrayList<Integer>();
        int count = 0;
        int y_0_init = 0;
        int y_1_init = rows - 1;
        int x0 = 0;
        int x1 = 0;
        for(Integer top_r: r_vals){
            int top_theta = theta_vals.get(count);
            try {
                x0 = (int) ((top_r - y_0_init * Math.sin(top_theta * Math.PI / 180.0))/Math.cos(top_theta * Math.PI /180.0));
                x1 = (int) ((top_r - y_1_init * Math.sin(top_theta * Math.PI / 180.0))/Math.cos(top_theta * Math.PI /180.0));

            }
            catch(Exception e){
                count = count + 1;
                continue;
            }
            try {
                if (top_theta >= 0 && top_theta < 45 || top_theta <= 180 && top_theta > 135){
                          Imgproc.line (
                            foreground_resized,                    //Matrix obj of the image
                            new Point(x0, y_0_init),        //p1
                            new Point(x1, y_1_init),       //p2
                            new Scalar(0, 255, 0),     //Scalar object for color
                            2                          //Thickness of the line
                         );
                         vert_line_r.add(top_r);
                         vert_line_theta.add(top_theta);
                }
                else {
                          Imgproc.line (
                            foreground_resized,                    //Matrix obj of the image
                            new Point(x0, y_0_init),        //p1
                            new Point(x1, y_1_init),       //p2
                            new Scalar(0, 255, 255),     //Scalar object for color
                            2                          //Thickness of the line
                         );
                          horz_line_r.add(top_r);
                          horz_line_theta.add(top_theta);
                }
            }
            catch(Exception e){
                count = count + 1;
                continue;
            }
            count = count + 1;
        }
        
        Imgcodecs.imwrite("lines.jpg", foreground_resized);

        
        // Get Points of Intersection
      
        ArrayList<Integer> intersect_points_rows = new ArrayList<Integer>();
        ArrayList<Integer> intersect_points_cols = new ArrayList<Integer>();
        int h_count = 0;
        int v_count = 0;
        for(Integer h_r: horz_line_r){
            Integer h_theta = horz_line_theta.get(h_count);
            for(Integer v_r: vert_line_r){
                Integer v_theta = vert_line_theta.get(v_count);
                double[][] inter = intersection(h_r, v_r, h_theta, v_theta);
    //            System.out.println(inter);
    //            System.out.println(Arrays.deepToString(inter));
     //          System.out.println(Arrays.toString(inter[0]));
     //          System.out.println("hi");
                if(inter[0][0] > 0.0 && inter[0][0] < cols && inter[1][0] > 0.0 && inter[1][0] < rows){
                    double[] ex_color = new double[]{255.0, 255.0, 0.0};
                    pooled.put((int)inter[1][0],(int)inter[0][0],ex_color);
                    intersect_points_rows.add((int)inter[1][0]);
                    intersect_points_cols.add((int)inter[0][0]);
                }
                v_count = v_count + 1;
            }
            h_count++;
            v_count = 0;
        }
        Imgcodecs.imwrite("pooled.jpg", pooled);

       
       int row_val = intersect_points_rows.size();
       Mat samples = new Mat(row_val,2, CvType.CV_32F);
       for(int i = 0; i < row_val; i++){
           samples.put(i, 0, intersect_points_rows.get(i));
           samples.put(i, 1, intersect_points_cols.get(i));
       }

       Mat labels = new Mat();

       int attempts = 3;
       Mat centers = new Mat();
       TermCriteria criteria = new TermCriteria(TermCriteria.EPS + 
            TermCriteria.MAX_ITER,100,0.1);
     // 3)  int clusterCount = 4;
     int clusterCount = 9;
       Core.kmeans(samples, clusterCount, labels, criteria, attempts, Core.KMEANS_PP_CENTERS, centers );
       
       int top_left_row = (int) Math.round(centers.get(2,1)[0]);
       int top_left_col = (int) Math.round(centers.get(2,0)[0]);
       int bottom_left_row = (int) Math.round(centers.get(3,1)[0]);;
       int bottom_left_col = (int) Math.round(centers.get(3,0)[0]);;
       int top_right_row = (int) Math.round(centers.get(1,1)[0]);;
       int top_right_col = (int) Math.round(centers.get(1,0)[0]);;
       int bottom_right_row = (int) Math.round(centers.get(0,1)[0]);;
       int bottom_right_col = (int) Math.round(centers.get(0,0)[0]);;
       
       System.out.println(top_left_row);
       System.out.println(top_left_col);
       System.out.println(bottom_left_row);
       System.out.println(bottom_left_col);
       System.out.println(top_right_row);
       System.out.println(top_right_col);
       System.out.println(bottom_right_row);
       System.out.println(bottom_right_col);
       
        // Draw Circles 
    /*    //Drawing a Circle
        Imgproc.circle (
           corner_img,                 //Matrix obj of the image
           new Point(top_left_row, top_left_col),    //Center of the circle
           3,                    //Radius
           new Scalar(0, 0, 255),  //Scalar object for color
           10                      //Thickness of the circle
        );
        Imgproc.circle (
           corner_img,                 //Matrix obj of the image
           new Point(bottom_left_row, bottom_left_col),    //Center of the circle
           3,                    //Radius
           new Scalar(0, 0, 255),  //Scalar object for color
           10                      //Thickness of the circle
        );
        Imgproc.circle (
           corner_img,                 //Matrix obj of the image
           new Point(top_right_row, top_right_col),    //Center of the circle
           3,                    //Radius
           new Scalar(0, 0, 255),  //Scalar object for color
           10                      //Thickness of the circle
        );
        Imgproc.circle (
           corner_img,                 //Matrix obj of the image
           new Point(bottom_right_row, bottom_right_col),    //Center of the circle
           3,                    //Radius
           new Scalar(0, 0, 255),  //Scalar object for color
           10                      //Thickness of the circle
        );*/
    // BIG CHANGE HERE!!!! FLIPPED CENTERS.gET()
    for(int i = 0; i < 9; i++){
        Imgproc.circle (
        corner_img,
        new Point((int) Math.round(centers.get(i,1)[0]),(int) Math.round(centers.get(i,0)[0])),
        3,
        new Scalar(0, 0, 255),
        10);
    }
          Imgcodecs.imwrite("hough_output.jpg", corner_img);

    
        for(int r = 0; r < rows; r++){
            for(int c = 0; c < cols; c++){
              double [] d = edges.get(r, c);
              if(d[0] > 0.0){
                  edge_point_rows.add(r);
                  edge_point_cols.add(c);
              }
            }
        }
        
        MatOfPoint2f src = new MatOfPoint2f(
                new Point(top_left_row, top_left_col),
                new Point(bottom_left_row, bottom_left_col),
                new Point(top_right_row, top_right_col),
                new Point(bottom_right_row, bottom_right_col)
            );


        MatOfPoint2f dst = new MatOfPoint2f(
                new Point(0, 0),
                new Point(450-1,0),
                new Point(0,450-1),
                new Point(450-1,450-1)      
                );
    
    Mat warpMat = Imgproc.getPerspectiveTransform(src,dst);
    //This is you new image as Mat
    Mat destImage = new Mat();
    Imgproc.warpPerspective(img_pt, destImage, warpMat, new Size(450,450));
               Imgcodecs.imwrite("pt_output.jpg", destImage);

        System.out.println(rows);
        System.out.println(cols);
        System.out.println("Image Loaded");
        
   // Color Decoding
   
    }
    
    public static double[][] intersection(Integer r1, Integer r2, Integer theta1, Integer theta2){
        double[][] A_vals = {{Math.cos(theta1*Math.PI/180.0),Math.sin(theta1*Math.PI/180.0)},{Math.cos(theta2*Math.PI/180.0),Math.sin(theta2*Math.PI/180.0)}};
        double[][] b_vals = {{r1},{r2}};
        Matrix A = new Matrix(A_vals);
        Matrix b = new Matrix(b_vals);
        Matrix x = A.solve(b);
        double [][] x_vals = x.getArray();
        return x_vals;
    }

    


    
}
