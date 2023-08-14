import os
from PIL import Image
from pylab import *
from numpy import *



def imlist(path):
    """
    The function imlist returns all the names of the files in 
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]

def get_imlist(path):
    """    Returns a list of filenames for 
        all jpg images in a directory. """
        
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


def compute_average(imlist):
    """    Compute the average of a list of images. """
    """ Post edit"""
    
    # open first image and make into array of type float


    averageim = array(Image.open(imlist[0]), 'f')   # makes the image matrix that of "floats"

    skipped = 0
    
    for imname in imlist[1:]:
        try: 
            averageim += array(Image.open(imname))
        except:
            print (imname , '+ ...skipped')
            skipped += 1

    averageim /= (len(imlist) - skipped)
    
    # return average as uint8
    return array(averageim, 'uint8')
    #return 333#averageim

    
def convert_to_grayscale(imlist):
    """    Convert a set of images to grayscale. """
    
    for imname in imlist:
        im = Image.open(imname).convert("L")
        im.save(imname)


def imresize(im,sz):
    """    Resize an image array using PIL. """
    pil_im = Image.fromarray(uint8(im))
    
    return array(pil_im.resize(sz))


def histeq(im,nbr_bins=256):
    """    Histogram equalization of a grayscale image. """
    
    # get image histogram
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    
    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1],cdf) #bins[:-1]  |  bins[1:]
    
    return im2.reshape(im.shape), cdf
    
    
def plot_2D_boundary(plot_range,points,decisionfcn,labels,values=[0]):
    """    Plot_range is (xmin,xmax,ymin,ymax), points is a list
        of class points, decisionfcn is a funtion to evaluate, 
        labels is a list of labels that decisionfcn returns for each class, 
        values is a list of decision contours to show. """
        
    clist = ['b','r','g','k','m','y'] # colors for the classes
    
    # evaluate on a grid and plot contour of decision function
    x = arange(plot_range[0],plot_range[1],.1)
    y = arange(plot_range[2],plot_range[3],.1)
    xx,yy = meshgrid(x,y)
    xxx,yyy = xx.flatten(),yy.flatten() # lists of x,y in grid
    zz = array(decisionfcn(xxx,yyy)) 
    zz = zz.reshape(xx.shape)
    # plot contour(s) at values
    contour(xx,yy,zz,values) 
        
    # for each class, plot the points with '*' for correct, 'o' for incorrect
    for i in range(len(points)):
        d = decisionfcn(points[i][:,0],points[i][:,1])
        correct_ndx = labels[i]==d
        incorrect_ndx = labels[i]!=d
        plot(points[i][correct_ndx,0],points[i][correct_ndx,1],'*',color=clist[i])
        plot(points[i][incorrect_ndx,0],points[i][incorrect_ndx,1],'o',color=clist[i])
    
    axis('equal')



class LKalman:
    def __init__(self):
        # u and v estimate states
        self.YH    = 0
        self.YDH   = 0
        self.XNTH  = 0
        self.YHy   = 0
        self.YDHy  = 0
        self.XNTHy = 0
        # covariances
        self.P11  = 0
        self.P12  = 0
        self.P13  = 0
        self.P22  = 0
        self.P23  = 0
        self.P33  = 0
        self.P11y = 0
        self.P12y = 0
        self.P13y = 0
        self.P22y = 0
        self.P23y = 0
        self.P33y = 0
        # Check email channels
        self.dtFreezeTimer=0
            
    def initialize(self,YH=0,YDH=0,XNTH=0,YHy=0,YDHy=0 ,XNTHy=0,P11=0,P12=0,P13=0,P22=0,P23=0,P33=36*36*1*2,P11y=0,P12y=0,P13y=0,P22y=0,P23y=0,P33y=36*36*1*2):
        self.YH    = YH   
        self.YDH   = YDH  
        self.XNTH  = XNTH 
        self.YHy   = YHy  
        self.YDHy  = YDHy 
        self.XNTHy = XNTHy
        self.P11  =  P11  
        self.P12  =  P12  
        self.P13  =  P13  
        self.P22  =  P22  
        self.P23  =  P23  
        self.P33  =  P33  
        self.P11y =  P11y 
        self.P12y =  P12y 
        self.P13y =  P13y 
        self.P22y =  P22y 
        self.P23y =  P23y 
        self.P33y =  P33y 
        
        
        
    def runKalman(self,uMeas, vMeas, measAvailable, deltaTime=0.03, noiseX=400, noiseY=400, accelMaxTargEst=1000):
        
        TS = deltaTime
        TS2= TS*TS
        TS3= TS2*TS
        TS4= TS3*TS
        TS5= TS4*TS

        
        PHISkx= accelMaxTargEst*accelMaxTargEst   
        M11=self.P11+TS*self.P12+.5*TS2*self.P13+TS*(self.P12+TS*self.P22+.5*TS2*self.P23)
        M11 = M11 +TS*(self.P12+TS*self.P22+.5*TS2*self.P23)
        M11= M11+.5*TS2*(self.P13+TS*self.P23+.5*TS2*self.P33)
        M11 = M11 +TS5*PHISkx/20.
        M12 = self.P12+TS*self.P22+.5*TS2*self.P23 
        M12 = M12 +TS*(self.P13+TS*self.P23+.5*TS2*self.P33)+TS4*PHISkx/8.
        M13= self.P13+TS*self.P23+.5*TS2*self.P33+PHISkx*TS3/6.
        M22= self.P22+TS*self.P23+TS*(self.P23+TS*self.P33)+PHISkx*TS3/3.
        M23= self.P23+TS*self.P33+.5*TS2*PHISkx
        M33= self.P33+PHISkx*TS
        
        M11y= self.P11y+TS*self.P12y+.5*TS2*self.P13y+TS*(self.P12y+TS*self.P22y+.5*TS2*self.P23y)
        M11y= M11y+.5*TS2*(self.P13y+TS*self.P23y+.5*TS2*self.P33y)+TS5*PHISkx/20.
        M12y= self.P12y+TS*self.P22y+.5*TS2*self.P23y+TS*(self.P13y+TS*self.P23y+.5*TS2*self.P33y)+TS4*PHISkx/8. 
        M13y=  self.P13y+TS*self.P23y+.5*TS2*self.P33y+PHISkx*TS3/6.
        M22y= self.P22y+TS*self.P23y+TS*(self.P23y+TS*self.P33y)+PHISkx*TS3/3.
        M23y= self.P23y+TS*self.P33y+.5*TS2*PHISkx
        M33y= self.P33y+PHISkx*TS
        
        
        
        SIGN2x = noiseX*noiseX  
        K1=M11/(M11+SIGN2x)
        K2=M12/(M11+SIGN2x) 
        K3=M13/(M11+SIGN2x)
        
        SIGN2y = noiseY*noiseY
        K1y=M11y/(M11y+SIGN2y)
        K2y=M12y/(M11y+SIGN2y) 
        K3y=M13y/(M11y+SIGN2y)
        
        self.P11=(1.-K1)*M11
        self.P12=(1.-K1)*M12
        self.P13=(1.-K1)*M13
        self.P22=-K2*M12+M22
        self.P23=-K2*M13+M23
        self.P33=-K3*M13+M33 
        
        self.P11y=(1.-K1y)*M11y  
        self.P12y=(1.-K1y)*M12y
        self.P13y=(1.-K1y)*M13y
        self.P22y=-K2y*M12y+M22y
        self.P23y=-K2y*M13y+M23y
        self.P33y=-K3y*M13y+M33y    
        
        # z_k*    " Already Defined " Measurement input
        
        # resolution calculation
        RES=  uMeas-self.YH-TS*self.YDH-.5*TS*TS*(self.XNTH);  # resolution = Xmeas - Xest
        RESy= vMeas-self.YHy-TS*self.YDHy-.5*TS*TS*(self.XNTHy); 
        
        # LKF Update
        if(measAvailable):
            self.dtFreezeTimer=0 
            self.YH = K1*RES+self.YH+TS*self.YDH+.5*TS*TS*(self.XNTH)
            self.YDH=(K2*RES+self.YDH)+TS*(self.XNTH);  
            self.XNTH=K3*RES+self.XNTH;
            
            self.YHy = K1y*RESy+ self.YHy+TS*self.YDHy+.5*TS*TS*(self.XNTHy)
            self.YDHy=(K2y*RESy+ self.YDHy)+TS*(self.XNTHy); 
            self.XNTHy=K3y*RESy+ self.XNTHy;
        
        # Constant velocity drift
        if(not (measAvailable) and self.dtFreezeTimer < 2 ):
            self.YH= self.YH+TS* self.YDH ; 
            self.YHy= self.YHy+TS*self.YDHy;
            self.dtFreezeTimer=self.dtFreezeTimer+1
            
            
	    
        '''uKalmanEstimate = self.YH
        vKalmanEstimate = self.YHy #'''
        
        
        targetLeadingGain = 2
        uKalmanEstimate = self.YH  + TS*TS*0.5* self.XNTH  * targetLeadingGain
        vKalmanEstimate = self.YHy + TS*TS*0.5* self.XNTHy * targetLeadingGain #'''
        
        
        return uKalmanEstimate, vKalmanEstimate
        
        
       
