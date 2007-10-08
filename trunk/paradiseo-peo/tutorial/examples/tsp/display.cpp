/* 
* <display.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sebastien Cahon, Alexandru-Adrian Tantar
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#include <iostream>
#include <fstream>

#include <X11/Xlib.h>

#include "display.h"
#include "node.h"
#include "opt_route.h"

#define BORDER 20
#define RATIO 0.5

#define screen_width 1024
#define screen_height 768

static const char * filename;

/* Computed coordinates */
static unsigned * X_new_coord, * Y_new_coord ;

/* this variable will contain the handle to the returned graphics context. */
static GC gc;
  
/* this variable will contain the pointer to the Display structure */
static Display* disp;

/* this variable will store the ID of the newly created window. */
static Window win;

static int screen;

/* Create a new backing pixmap of the appropriate size */

  /* Best tour */
  /*
  gdk_gc_set_line_attributes (gc, 2,  GDK_LINE_ON_OFF_DASH, GDK_CAP_NOT_LAST, GDK_JOIN_MITER) ;

  gdk_gc_set_foreground  (gc, & color_green) ;      

  for (int i = 0 ; i < (int) numNodes ; i ++) {

    gdk_draw_line (pixmap, gc,
		   X_new_coord [opt_route [i]],
		   Y_new_coord [opt_route [i]],
		   X_new_coord [opt_route [(i + 1) % numNodes]],
		   Y_new_coord [opt_route [(i + 1) % numNodes]]);
    
		   }*/

void openMainWindow (const char * __filename) {

  filename = __filename;

  /* Map */
  int map_width = (int) (X_max - X_min);
  int map_height = (int) (Y_max - Y_min);
  int map_side = std :: max (map_width, map_height);
  
  /* Calculate the window's width and height. */
  int win_width = (int) (screen_width * RATIO * map_width / map_side);
  int win_height = (int) (screen_height * RATIO * map_height / map_side);

  /* Computing the coordinates */
  X_new_coord = new unsigned [numNodes];
  Y_new_coord = new unsigned [numNodes];

  for (unsigned i = 0; i < numNodes; i ++) {
    X_new_coord [i] = (unsigned) (win_width * (1.0 - (X_coord [i] - X_min) / map_width) + BORDER);
    Y_new_coord [i] = (unsigned) (win_height * (1.0 - (Y_coord [i] - Y_min) / map_height) + BORDER);
  }
  
  /* Initialisation */
  XGCValues val ;
  
  disp = XOpenDisplay (NULL) ;
  screen = DefaultScreen (disp) ;
  win = XCreateSimpleWindow (disp, RootWindow (disp, screen), 0, 0, win_width + 2 * BORDER, win_height + 2 * BORDER, 2, BlackPixel (disp, screen), WhitePixel (disp, screen)) ;
  val.foreground = BlackPixel(disp, screen) ;
  val.background = WhitePixel(disp, screen) ;
  gc = XCreateGC (disp, win, GCForeground | GCBackground, & val) ; 

  XMapWindow (disp, win) ;
  XFlush (disp) ;

  while (true) {
    XClearWindow (disp, win) ;

    /* Vertices as circles */
    for (unsigned i = 1 ; i < numNodes ; i ++)
      XDrawArc (disp, win, gc, X_new_coord [i] - 1, Y_new_coord [i] - 1, 3, 3, 0, 364 * 64) ;
    
    /* New tour */
    std :: ifstream f (filename);
    if (f) {
      Route route;
      f >> route;
      f.close ();
      
      for (int i = 0; i < (int) numNodes; i ++) 
	XDrawLine (disp, win, gc,     
		   X_new_coord [route [i]],
		 Y_new_coord [route [i]],
		   X_new_coord [route [(i + 1) % numNodes]],
		   Y_new_coord [route [(i + 1) % numNodes]]);  
    }
    XFlush (disp) ;    
    sleep (1) ;
  }
}
