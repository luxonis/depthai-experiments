import pygame
from pygame.locals import *
import math

from OpenGL.GL import *
from OpenGL.GLU import *

lastPosX = 0;
lastPosY = 0;
zoomScale = 1.0;
dataL = 0;
xRot = 0;
yRot = 0;
zRot = 0;


#verticies = (
#    (0.97032104,0.22256307, 0.09456615),
#    (0.97020411, -0.21309532,  0.11530124),
#    (0.99975873 , 0.01945705, -0.01019372),
#    (0.94979962 , 0.16702298, -0.26454489),
#    (0.95420808 ,-0.16182911, -0.25159151),
#    (0.107, -0.038 ,0.008),
#    (0.109, 0.039 , 0.008))
#    #5, 6
#edges = (
#    (5,0),
#    (5,1),
#    (5,2),
#    (5,3),
#    (5,4)
#    )
#def background_pad():
    
#def landmark_visualizer(left_landmarks, right_landmarks,cameras):
#    glLineWidth(1.5)
#    glBegin(GL_LINES)
##    if flag:
##        glColor3f(0.0,1.0,0.0);
##    else:
##        glColor3f(0.0,0.0,1.0)
#
#    glColor3f(0.0,1.0,0.0);
#    for landmark in left_landmarks:
#        glVertex3fv(cameras[0])
#        glVertex3fv(landmark)
#
#    glColor3f(0.0,0.0,1.0)
#    for landmark in right_landmarks:
#        glVertex3fv(cameras[1])
#        glVertex3fv(landmark)
#    glEnd()
#    # print(repr(verticies[0][0])+','+repr(verticies[0][1]));
#
#    glPointSize(3.0)
#    glBegin(GL_POINTS);
#    glColor3f(1.0,0.0,0.0);
#    for i in range(len(left_landmarks)):
#        glVertex3f(left_landmarks[i][0], left_landmarks[i][1], left_landmarks[i][2]);
#    glEnd();
#
#    glPointSize(3.0)
#    glBegin(GL_POINTS);
#    glColor3f(1.0,0.0,0.0);
#    for i in range(len(right_landmarks)):
#        glVertex3f(right_landmarks[i][0], right_landmarks[i][1], right_landmarks[i][2]);
#    glEnd();
#
#    glPointSize(3.0)
#    glBegin(GL_POINTS);
#    glColor3f(1.0,0.0,0.0);
#    for i in range(len(cameras)):
#        glVertex3f(cameras[i][0], cameras[i][1], cameras[i][2]);
#    glEnd();

def landmark_visualizer(landmarks,cameras, left_landmarks, right_landmarks):
    glLineWidth(1.5)
    glBegin(GL_LINES)

    glColor3f(0.0,1.0,0.0);
    for landmark in landmarks:
        glVertex3fv(cameras[0])
        glVertex3fv(landmark)

    glColor3f(0.0,0.0,1.0)
    for landmark in landmarks:
        glVertex3fv(cameras[1])
        glVertex3fv(landmark)
    glEnd()

    glPointSize(3.0)
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);
    for i in range(len(landmarks)):
        glVertex3f(landmarks[i][0], landmarks[i][1], landmarks[i][2]);
    glEnd();

    glPointSize(3.0)
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);
    for i in range(len(cameras)):
        glVertex3f(cameras[i][0], cameras[i][1], cameras[i][2]);
    glEnd();
    
    glLineWidth(1.5)
    glBegin(GL_LINES)

    glColor3f(0.0,1.0,0.0);
    for landmark in left_landmarks:
        glVertex3fv(cameras[0])
        glVertex3fv(landmark)

    glColor3f(0.0,0.0,1.0)
    for landmark in right_landmarks:
        glVertex3fv(cameras[1])
        glVertex3fv(landmark)
    glEnd()

    glPointSize(3.0)
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);
    for i in range(len(left_landmarks)):
        glVertex3f(left_landmarks[i][0], left_landmarks[i][1], left_landmarks[i][2]);
    glEnd();

    glPointSize(3.0)
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);
    for i in range(len(right_landmarks)):
        glVertex3f(right_landmarks[i][0], right_landmarks[i][1], right_landmarks[i][2]);
    glEnd();

    glPointSize(3.0)
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);
    for i in range(len(cameras)):
        glVertex3f(cameras[i][0], cameras[i][1], cameras[i][2]);
    glEnd();


def mouseMove(event):
    global lastPosX, lastPosY, zoomScale, xRot, yRot, zRot;

    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
        glScaled(1.05, 1.05, 1.05);
    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
        glScaled(0.95, 0.95, 0.95);

    if event.type == pygame.MOUSEMOTION:
        x, y = event.pos;
        dx = x - lastPosX;
        dy = y - lastPosY;

        mouseState = pygame.mouse.get_pressed();
        if mouseState[0]:

            modelView = (GLfloat * 16)()
            mvm = glGetFloatv(GL_MODELVIEW_MATRIX, modelView)

            temp = (GLfloat * 3)();
            temp[0] = modelView[0]*dy + modelView[1]*dx;
            temp[1] = modelView[4]*dy + modelView[5]*dx;
            temp[2] = modelView[8]*dy + modelView[9]*dx;
            norm_xy = math.sqrt(temp[0]*temp[0] + temp[1]*temp[1] + temp[2]*temp[2]);
            glRotatef(math.sqrt(dx*dx+dy*dy), temp[0]/norm_xy, temp[1]/norm_xy, temp[2]/norm_xy);

        lastPosX = x;
        lastPosY = y;



def initialize_OpenGL():
    pygame.init()

    display = (300,300)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL, RESIZABLE)

    gluPerspective(45, (1.0*display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0,0.0, -5)


def start_OpenGL(landmarks, cameras, left_landmarks, right_landmarks):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        mouseMove(event);

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    landmark_visualizer(landmarks,cameras, left_landmarks, right_landmarks)
    pygame.display.flip()
    pygame.time.wait(1)

#def start_OpenGL(left_landmarks,right_landmarks, cameras):
#    for event in pygame.event.get():
#        if event.type == pygame.QUIT:
#            pygame.quit()
#            quit()
#        mouseMove(event);
#
#    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
#    landmark_visualizer(left_landmarks,right_landmarks,cameras)
#    pygame.display.flip()
#    pygame.time.wait(1)
