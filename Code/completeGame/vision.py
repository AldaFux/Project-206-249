import numpy as np
import cv2
# os.chdir("C:/Users/patri/Dropbox/UCB/Project-206-249-Padde/niko/JoinTheDisciplines")
class Image():
    def __init__(self):
        cv2.ocl.setUseOpenCL(False)
        self.calibrated = False
        self.orig = []
        self.transform = []
        self.gray = []
        self.adap_thresh = []
        self.left_region = np.zeros([2,2])
        self.width = 0
        self.height = 0
        self.sidelength = 0
        self.moving_min_area = 500.0
        self.token_min_area = 1200.0
        self.tile_min_area = 300.0
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(0)
        return self
        
    def __exit__(self, exception, value, traceback):
        self.cap.release()
        
    def __record_frame(self):
        recording, self.orig = self.cap.read()
        if not recording:
            raise Exception('ERROR: Cannot access camera.')
        if self.calibrated:
            self.transform = cv2.warpPerspective(self.orig,self.transformationMatrix,(self.sidelength,self.sidelength))
        else:
            # Set dimensions of transformed image
            self.height, self.width, ch = self.orig.shape
            if (self.width > self.height):
                self.sidelength = self.height
            else:
                self.sidelength = self.width
        
    def __is_moving(self,img):
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        kernel = np.ones((5,5),np.uint8)
        fgmask = self.fgbg.apply(img)
        ret, bw = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        
        opening, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # loop over the contours
        for cnt in contours:
            # if the contour is too small, ignore it
            if cv2.contourArea(cnt) < self.moving_min_area:
                continue
            else:
                return True
        print('No movement detected.')
        return False
            
    def show_frame(self):
        cv2.imshow('Frame captured',self.orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows
    
    def show_transform(self):
        cv2.imshow('Transformed board',self.transform)
        cv2.waitKey(0)
        cv2.destroyAllWindows
                        
    def calibrate(self):
        # Record two images, so that the camera adjusts shutter time
        self.__record_frame()
        self.__record_frame()
        #self.show_frame() #Debug
        
        # Convert image to binary
        gray = cv2.cvtColor(self.orig,cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 123, 50)
        
        # Find the contours in the binary image        
        cnt_img, contours, hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        # Find inner contour which should be the middle field
        highest_hierarchy_level = -1
        found_middle_field = False
        for j in range(0,len(contours)):
            cnt = contours[j]
            hull = cv2.convexHull(cnt)
            area = cv2.contourArea(hull)
            # If area of enclosing contour is too small, skip it
            if (area<self.tile_min_area):
                continue
            # Check if the hierarchy is smaller than the last one.
            hier = hierarchy[0][j]
            if hier[3]>highest_hierarchy_level:
                highest_hierarchy_level = hier[3]
                inner_cnt = j
                found_middle_field = True
        
        if found_middle_field==False:
            raise Exception ('Was not able to detect the field.')
        # Fit a rectangle in the detected middle field
        cnt = contours[inner_cnt]
        x,y,w,h = cv2.boundingRect(cnt)
        # Save the lower right corner of the region left from the field
        # Here, the sign indicating if the player wants to start or not will be detected
        left_region_x = x-1.5*w
        left_region_y = y+2.5*h
        if (left_region_x<0):
            raise Exception ('Not able to see region left of the field. Adjust camera position.')
        if (left_region_y>self.height):
            left_region_y = self.height
        self.left_region[1,:] = [left_region_x,left_region_y]
        # Calculate corner locations in source image from rectangle corners
        urSrc = [x+w,y]
        lrSrc = [x+w,y+h]
        llSrc = [x,y+h]
        ulSrc = [x,y]
        cornersSrc = np.float32([urSrc, lrSrc, llSrc, ulSrc])
        
        # Define corner positions in destination image
        urDst = [0.65*self.sidelength,0.35*self.sidelength]
        lrDst = [0.65*self.sidelength,0.65*self.sidelength]
        llDst = [0.35*self.sidelength,0.65*self.sidelength]
        ulDst = [0.35*self.sidelength,0.35*self.sidelength]
        cornersDst = np.float32([urDst, lrDst, llDst, ulDst])
        
        # Transform the image    
        self.transformationMatrix = cv2.getPerspectiveTransform(cornersSrc,cornersDst)
        self.transform = cv2.warpPerspective(self.orig,self.transformationMatrix,(self.sidelength,self.sidelength))
        self.calibrated = True
        
    def detect_sign(self,emptyTiles):
        # emptyTiles should list with empty tiles
        emptyTiles_shift = list()
        emptyTiles_shift = [x-1 for x in emptyTiles]
        
        # print (emptyTiles)
        
        # shape detection from
        # http://stackoverflow.com/questions/31974843/detecting-lines-and-shapes-in-opencv-using-python
        # and http://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
        
        self.__record_frame()
        
        while self.__is_moving(self.transform):
            cv2.imshow('Transformed board',self.transform)
            print("Something is moving.")
            cv2.waitKey(500)
        
        cv2.imshow('Transformed board',self.transform)

        gray = cv2.cvtColor(self.transform,cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 123, 50)
        
        # res = self.transform.copy() # DEBUG
        
        # Define the regions of the tiles in the image
        y11 = x11 = 0
        y12 = x12 = int(0.35*self.sidelength)
        y21 = x21 = int(0.35*self.sidelength)
        y22 = x22 = int(0.65*self.sidelength)
        y31 = x31 = int(0.65*self.sidelength)
        y32 = x32 = int(self.sidelength)
        # Create a Region of Interest for each tile
        tile_ll = bw[y31:y32,x11:x12]
        tile_lm = bw[y31:y32,x21:x22]
        tile_lr = bw[y31:y32,x31:x32]
        tile_ml = bw[y21:y22,x11:x12]
        tile_mm = bw[y21:y22,x21:x22]
        tile_mr = bw[y21:y22,x31:x32]
        tile_ul = bw[y11:y12,x11:x12]
        tile_um = bw[y11:y12,x21:x22]
        tile_ur = bw[y11:y12,x31:x32]
        tiles = [tile_ll,tile_lm,tile_lr,tile_ml,tile_mm,tile_mr,tile_ul,tile_um,tile_ur]
        # print('The length of the list of tiles is {}'.format(len(tiles)))
        # print('The length of the list of tiles[0] is {}'.format(len(tiles[0])))
        
        
        # Define the regions where the center of the token needs to be
        yc11 = xc11 = 0
        yc12 = xc12 = int(0.25*self.sidelength)
        yc21 = xc21 = int(0.45*self.sidelength)
        yc22 = xc22 = int(0.55*self.sidelength)
        yc31 = xc31 = int(0.75*self.sidelength)
        yc32 = xc32 = int(self.sidelength)
        # Create lower and upper bounds for x and y values for each tile
        center_region_ll = np.array([[xc11,yc31],[xc12,yc32]])
        center_region_lm = np.array([[xc21,yc31],[xc22,yc32]])
        center_region_lr = np.array([[xc31,yc31],[xc32,yc32]])
        center_region_ml = np.array([[xc11,yc21],[xc12,yc22]])
        center_region_mm = np.array([[xc21,yc21],[xc22,yc22]])
        center_region_mr = np.array([[xc31,yc21],[xc32,yc22]])
        center_region_ul = np.array([[xc11,yc11],[xc12,yc12]])
        center_region_um = np.array([[xc21,yc11],[xc22,yc12]])
        center_region_ur = np.array([[xc31,yc11],[xc32,yc12]])
        center_regions = np.array([center_region_ll,center_region_lm,center_region_lr,center_region_ml,center_region_mm,center_region_mr,center_region_ul,center_region_um,center_region_ur])
        # Indexing: center_regions[tile,bound upper/lower,x/y]
        
        for i in emptyTiles_shift:
            tile = tiles[i]
            print('Looking for token in tile {}'.format(i+1))
            # cv2.imshow('tile',tile)
            # print('The sidelength of one tile is {}'.format(tile.shape))
            
            # RETR_EXTERNAL
            # If you use this flag, it returns only extreme outer flags. All child contours are left behind. 
            # http://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
            cnt_img, contours, hierarchy = cv2.findContours(tile,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            tokens = list(contours)
            deleteCounter = 0
            for j in range(0,len(contours)):
                cnt = contours[j]
                hull_indices = cv2.convexHull(cnt,returnPoints = False)
                hull = cv2.convexHull(cnt)
                area = cv2.contourArea(hull)
                # Check if area covered by contour is big enough to be a token
                if (area<self.token_min_area):
                    tokens.pop(j-deleteCounter)
                    deleteCounter += 1
                    continue
                # Check if center of mass of the area is in the center region of the tile
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                if not ((center_regions[i,0,0] < cx) and (cx < center_regions[i,1,0]) and (center_regions[i,0,1] < cy) and (cy < center_regions[i,1,1])):
                    # Center of mass of the contour is NOT in middle region of the tile
                    continue
                    
                # cv2.drawContours(tile,cnt,-1,(255,0,0),3)
                defects = cv2.convexityDefects(cnt, hull_indices)
                distances = np.array(defects[:,:,3])
                # print('The distances in tile {} are:'.format(i))
                # print(distances)
                # print('The area of the token in tile {} is {}'.format(i,area))
                mean = np.mean(distances[1:])
                # xOffset = int(((0.4*(i%3>0)) + (0.3*(i%3>1)))*self.sidelength)
                # yOffset = int(((0.4*((i//3)<2)) + (0.3*((i//3)<1)))*self.sidelength)
                # offset = (xOffset,yOffset)
                if 1000<mean:
                    # CROSS
                    # x,y,w,h = cv2.boundingRect(cnt)
                    # cv2.line(res,(x+xOffset,y+yOffset),(x+w+xOffset,y+h+yOffset),(60,200,255),2)
                    # cv2.line(res,(x+xOffset,y+h+yOffset),(x+w+xOffset,y+yOffset),(60,200,255),2)
                    # cv2.line(tile,(x,y),(x+w,y+h),(60,200,255),2)
                    # cv2.line(tile,(x,y+h),(x+w,y),(60,200,255),2)
                    print('Found cross in tile {}'.format(i+1))
                    
                    return [True,'X', i+1]
                else:
                    # CIRCLE
                    # (x,y),radius = cv2.minEnclosingCircle(cnt)
                    # center = (int(x+xOffset),int(y+yOffset))
                    # center2 = (int(x),int(y))
                    # radius = int(radius)
                    # cv2.circle(res,center,radius,(200,60,255),2)
                    # cv2.circle(tile,center2,radius,(200,60,255),2)
                    print('Found circle in tile {}'.format(i+1))
                    return [True,'O', i+1]
                    
        cv2.waitKey(500)
        return [False,'.',-1]
        
    def detect_first_move(self):
        if not self.calibrated:
            raise Exception ('Calibrate the camera before detecting the first move.')
        self.__record_frame()
        roi = self.orig[self.left_regi`on[0,1]:self.left_region[1,1],self.left_region[0,0]:self.left_region[1,0]]
        
        while self.__is_moving(roi):
            print("Something is moving.")
            cv2.waitKey(500)
        
        gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 123, 50)
        
            
        # RETR_EXTERNAL
        # If you use this flag, it returns only extreme outer flags. All child contours are left behind. 
        # http://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        cnt_img, contours, hierarchy = cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        for j in range(0,len(contours)):
            cnt = contours[j]
            hull_indices = cv2.convexHull(cnt,returnPoints = False)
            hull = cv2.convexHull(cnt)
            area = cv2.contourArea(hull)
            # Check if area covered by contour is big enough to be a token
            if (area<self.tile_min_area/2):
                continue
                
            # cv2.drawContours(tile,cnt,-1,(255,0,0),3)
            defects = cv2.convexityDefects(cnt, hull_indices)
            distances = np.array(defects[:,:,3])
            # print('The distances in tile {} are:'.format(i))
            # print(distances)
            # print('The area of the token in tile {} is {}'.format(i,area))
            mean = np.mean(distances[1:])
            # xOffset = int(((0.4*(i%3>0)) + (0.3*(i%3>1)))*self.sidelength)
            # yOffset = int(((0.4*((i//3)<2)) + (0.3*((i//3)<1)))*self.sidelength)
            # offset = (xOffset,yOffset)
            if 800<mean:
                # CROSS
                # x,y,w,h = cv2.boundingRect(cnt)
                # cv2.line(res,(x+xOffset,y+yOffset),(x+w+xOffset,y+h+yOffset),(60,200,255),2)
                # cv2.line(res,(x+xOffset,y+h+yOffset),(x+w+xOffset,y+yOffset),(60,200,255),2)
                # cv2.line(tile,(x,y),(x+w,y+h),(60,200,255),2)
                # cv2.line(tile,(x,y+h),(x+w,y),(60,200,255),2)
                print('Found cross.')
                return [True,'X']
            else:
                # CIRCLE
                # (x,y),radius = cv2.minEnclosingCircle(cnt)
                # center = (int(x+xOffset),int(y+yOffset))
                # center2 = (int(x),int(y))
                # radius = int(radius)
                # cv2.circle(res,center,radius,(200,60,255),2)
                # cv2.circle(tile,center2,radius,(200,60,255),2)
                print('Found circle.')
                return [True,'O']
                
        cv2.waitKey(500)
        return [False,'.']