# Detect and Recognize Car License Plate from a video in real time:

 Recognizing a Car License Plate is a very important task for a camera surveillance-based security system. We can extract the license plate from an image using some computer vision techniques and then we can use Optical Character Recognition to recognize the license number.

## Approach:

- Find all the contours in the image.
- Find the bounding rectangle of every contour.
- Compare and validate the sides ratio and area of every bounding rectangle with an average license plate.
- Apply image segmentation in the image inside the validated contour to find characters in it.
- Recognize characters using an OCR.

## Methodology:

- To reduce the noise we need to blur the input Image with Gaussian Blur and then convert it to grayscale.
  ![image1](https://media.geeksforgeeks.org/wp-content/uploads/20200326001440/gray1.jpg)
- Find vertical edges in the image.
  ![image2](https://media.geeksforgeeks.org/wp-content/uploads/20200326000832/edge1.jpg)
- To reveal the plate we have to binarize the image. For this apply Otsu’s Thresholding on the vertical edge image.
![image3](https://media.geeksforgeeks.org/wp-content/uploads/20200326001732/threshold2.jpg)
- Apply Closing Morphological Transformation on the thresholded image. Closing is useful to fill small black regions between white regions in a thresholded image. It reveals the rectangular white box of license plates.
![image4](https://media.geeksforgeeks.org/wp-content/uploads/20200326002340/morph.jpg)
- To detect the plate we need to find contours in the image. It is important to binarize and morph the image before finding contours so that it can find a more relevant and less number of contours in the image.
![image5](https://media.geeksforgeeks.org/wp-content/uploads/20200326010304/contour.jpg)
- Now find the minimum area rectangle enclosed by each of the contours and validate their side ratios and area. We have defined the minimum and maximum area of the plate as 4500 and 30000 respectively.
- Now find the contours in the validated region and validate the side ratios and area of the bounding rectangle of the largest contour in that region. After validating you will get a perfect contour of a license plate. Now extract that contour from the original image.
- To recognize the characters on the license plate precisely, we have to apply image segmentation. The first step is to extract the value channel from the HSV format of the plate’s image.
- Now apply adaptive thresholding on the plate’s value channel image to binarize it and reveal the characters. The image of the plate can have different lighting conditions in different areas, in that case, adaptive thresholding can be more suitable to binarize because it uses different threshold values for different regions based on the brightness of the pixels in the region around it.
- After binarizing apply bitwise not operation on the image to find the connected components in the image so that we can extract character candidates.
- Construct a mask to display all the character components and then find contours in the mask. After extracting the contours take the largest one, find its bounding rectangle and validate side ratios.
- After validating the side ratios find the convex hull of the contour and draw it on the character candidate mask.
- Now find all the contours in the character candidate mask and extract those contour areas from the plate’s value thresholded image, you will get all the characters separately.
