
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import glob
from sklearn.decomposition import PCA
from time import time

def detect_features(image):
    descriptor = cv2.xfeatures2d.SIFT_create()
    (keypoints, features) = descriptor.detectAndCompute(image, None)
    
    return (keypoints, features)

def match_keypoints(f1, f2,method):


    if method == 'bf':
      matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
      matched_points = matcher.match(f1,f2)
      #matched_points = matcher.knnMatch(f1,f2,2)
    else:
      FLAN_INDEX_KDTREE = 0
      index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
      search_params = dict (checks=50)
      flann = cv2.FlannBasedMatcher(index_params, search_params)
      matched_points = flann.knnMatch(f1, f2,k=2)

    #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    #bf = createMatcher(cv2.NORM_L2, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    #matched_points = bf.knnMatch(f1, f2, 2)
    #print("Raw matches (knn):", len(rawMatches))

    return matched_points

def display(image,caption = ''):
    plt.figure(figsize = (5,10))
    plt.title(caption)
    plt.imshow(image)
    plt.show()
    
def FindAndMatchDescriptors(img1, img2, method = 'bf'):
    gray_R = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_Q = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
    
    #gray_Q = cv2.equalizeHist(gray_Q)
    #gray_R = cv2.equalizeHist(gray_R)

    
    #k = 5
    #gray_Q = cv2.GaussianBlur(gray_Q,(k,k),1)
    #gray_R = cv2.GaussianBlur(gray_R,(k,k),1)
    #gray_Q = cv2.adaptiveThreshold(gray_Q, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #gray_R = cv2.adaptiveThreshold(gray_R, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    pointsR, featuresR = detect_features(gray_R)
    pointsQ, featuresQ = detect_features(gray_Q)

    #featuresR = apply_PCA(featuresR,32)
    #featuresQ = apply_PCA(featuresQ,32)
    
    matched_points = match_keypoints(featuresR, featuresQ,method)
    
    
    #best_points = int(len(matched_points) * 0.5)
    #matched_points = matched_points[:best_points]
    
    #print('Matched Points: ', len(matched_points))
    if method == 'bf':
      matched_points.sort(key=lambda x: x.distance, reverse=False)
      result = cv2.drawMatches(imageR,pointsR,imageQ,pointsQ,matched_points,None)
    else:
      #matched_points.sort(key=lambda x: x[0].distance, reverse=False)
      result = cv2.drawMatchesKnn(imageR,pointsR,imageQ,pointsQ,matched_points,None, flags = 2)
    
    return (result,pointsR,pointsQ,featuresR,featuresQ, matched_points)

def estimateTransformation(p1, p2, f1, f2, matches, method = 'bf'):
    
    p1 = np.float32([kp.pt for kp in p1])
    p2 = np.float32([kp.pt for kp in p2])

    if len(matches) > 4:        
        if method == 'bf':
          pts1 = np.float32([p1[m.queryIdx] for m in matches])
          pts2 = np.float32([p2[m.trainIdx] for m in matches])
        else:
          pts1 = np.float32([p1[m[0].queryIdx] for m in matches])
          pts2 = np.float32([p2[m[0].trainIdx] for m in matches])
    
        (H, mask) = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5)
        
        return (H, mask)
    else:
        return None

def findInliers(mask):
    in_points = 0
    for i in range(len(mask)):
        if mask[i] == 1:
            in_points = in_points+1
    
    return in_points

# Apply PCA on given dataset 

def apply_PCA(data,N=32):
    
    pca = PCA(N)
    h,w = data.shape
        
    transformed = pca.fit_transform(data)
    
    inverse = pca.inverse_transform(transformed)
    compressed = (np.dstack((inverse))).astype(np.uint8)
    compressed = compressed.reshape(h,w)
        
    return compressed


if __name__ == '__main__':
  
  #path = '/content/drive/MyDrive/A2_smvs/landmarks/Query/*.jpg' 
  path = '/content/drive/MyDrive/WineLabel/Wine Labels/*.jpg' 
  imagePaths = glob.glob(path)
  imagePaths = np.sort(imagePaths)
  
  imageQ = cv2.imread('/content/drive/MyDrive/WineLabel/Query Images/IMG_2.jpg')
  imageQ = cv2.resize(imageQ,(0,0),fx = 0.5,fy = 0.5)
  #display(imageQ, 'Query Image')
  #imageQ = cv2.imread('/Users/eapplestroe/Downloads/A2_smvs/book_covers/Query/041.jpg')
  
  counter = 1
  all_inliers = []
  all_images = []

  method = 'bf'   # bf or flann
  print('Process Started using '+method+' based matcher...')

  start = time()
  for image in imagePaths:
      print('Processing Image ',counter)
      imageR = cv2.imread(image)
      imageR = cv2.resize(imageR,(0,0),fx = 0.3,fy = 0.3)
      #(result,pointsR,pointsQ,featuresR,featuresQ,matched_points) = FindAndMatchDescriptors(imageR, imageQ)
      (result,pointsR,pointsQ,featuresR,featuresQ,matched_points) = FindAndMatchDescriptors(imageR, imageQ, method)
      #display(result, 'Matching Using L2 Normalization')

      matrix,mask = estimateTransformation(pointsR, pointsQ, featuresR, featuresQ, matched_points, method)
      inliers = findInliers(mask)
      #print('Inliers(TP) and Outliers(FP) for Image'+str(counter)+': ',inliers,len(mask)-inliers,'respectively')

      result = cv2.warpPerspective(imageR, matrix, (imageQ.shape[1], imageQ.shape[0]))
      #display(result, 'Extracted Object')

      #draw_outline(imageR.copy(),imageQ.copy(),matrix)
      
      counter = counter+1
      all_inliers.append(inliers)
      all_images.append(image)


  end = time()


  index = all_inliers.index(np.max(all_inliers))
  image1 = cv2.imread(all_images[index])
  #display(image,'1st Best Match')
  all_inliers[index] = 0

  index = all_inliers.index(np.max(all_inliers))
  image2 = cv2.imread(all_images[index])
  #display(image, '2nd Best Match')
  all_inliers[index] = 0

  index = all_inliers.index(np.max(all_inliers))
  image3 = cv2.imread(all_images[index])
  #display(image,'3rd Best Match')
  all_inliers[index] = 0  

  fig, (ax0,ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=4, figsize=(15,10), constrained_layout=False)
  ax0.imshow(imageQ)
  ax0.set_xlabel("Query Image", fontsize=14)
  ax1.imshow(image1)
  ax1.set_xlabel("1st Match", fontsize=14)
  ax2.imshow(image2)
  ax2.set_xlabel("2nd Match", fontsize=14)
  ax3.imshow(image3)
  ax3.set_xlabel("3rd Match", fontsize=14)
  
  print('Time taken using SIFT and '+method+' based matcher in seconds = ',abs(start-end))
      
  
    
    
    
    
    
