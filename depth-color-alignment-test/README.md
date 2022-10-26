# Depth-color alignment test
## Controls
| key | action |
| :--- | :--- |
| `q` | quit |
| `s` | select the elevated plane area on the RGB image |

## Usage
The camera should look on a scene with ground plane and elevation plane of different colors.

Press the `s` key to select the elevated area. Check if the segmentation looks correct. 

The running average of the error ratio will be displayed. 

## How it works
When the user makes a selection the average depth of the elevated and ground plane is computed. Based on that measurement the depth image is into _elevated area_ (pixels between _top padding_ and _mid depth_) and _ground area_ (pixels between _mid depth_ and _bottom padding_).

Similarly the color image is segmented based on the measured hue on the elevated and ground area.

```                                                                                                                                                                                  
                 ┌────────┐                                     
                 │ camera │                                     
                 └────────┘                                     
                    /  \                                        
                   /    \                                       
                  /      \                                      
                 /        \                                     
                /          \                                    
               /            \                                   
              /              \                                  
             /                \                                 
            /                  \                                
 ----------/--------------------\------------- top padding      
          /                      \                              
 ========/=======▒▒▒▒▒▒▒▒▒▒=======\=========== elevated plane   
        /                          \                            
       /                            \                           
 -----/------------------------------\-------- mid depth        
     /                                \                         
    /                                  \                        
 ██████████████████████████████████████████=== ground plane     
                                                                
 --------------------------------------------- bottom padding   

```

After segmentation the error ratio is computed.

![test](demo.png)