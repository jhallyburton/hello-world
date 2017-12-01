''' -*- Mode: Python; tab-width: 4; indent-tabs-mode: nil -*- '''

from __future__ import division

"""
Works on Python v2, NOT TESTED ON Python v3.

Based on an earlier post:
http://osdir.com/ml/python.matplotlib.devel/2005-07/msg00028.html

Main improvement: added an additional raw_limit element to specify
how many degrees of arc the gauge should subtend, defaulting to 180
if not supplied, i.e., upward compatible from previous instances.
Experience has shown 120 degrees looks better than 180.
-- John Hallyburton, 4/15/08

The Gauge widget draws a semi-circular gauge. You supply raw_limits,
shaded regions, names and the current value, and invoke it like this:


    from pylab import figure, show

    raw_value = -4.0
    raw_limits = [-1.0,1.0,1,0.1, 180] # Last element is arc length of gauge in degrees
    raw_zones = [[-1.0,0.0,'r'],[0.0,0.5,'y'],[0.5,1.0,'g']]
    attribute_name = "Rx MOS (24h)"

    file_name    = 'gauge.png'
    resolution   = 72

    graph_height = 1.6
    graph_width  = 2.8
    fig_height   = graph_height
    fig_width    = graph_width

    x_length     = graph_width
    y_length     = graph_height

    fig = figure( figsize=(fig_width, fig_height) )
  
    rect = [(0.0/fig_width), (0.2/fig_height),
            (graph_width/fig_width), (graph_height/fig_height)]

    gauge = Gauge( raw_value,
                   raw_limits, raw_zones,
                   attribute_name, field_names,
                   file_name, resolution,
                   x_length, y_length,
                   fig, rect,
                   xlim=( -0.1, graph_width+0.1 ),
                   ylim=( -0.4, graph_height+0.1 ),
                   xticks=[],
                   yticks=[],
                   )

    gauge.set_axis_off()
    fig.add_axes(gauge)

    show()                                                                                 

"""
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import math
import types

from math import pi

  
class Gauge(Axes):
    def __init__(self, raw_value, raw_limits, raw_zones, attribute_name, field_names, file_name, resolution, x_length, y_length, *args, **kwargs):
        Axes.__init__(self, *args, **kwargs)

        #Perform Checking
        if( raw_limits[0] == raw_limits[1] ):
            raise ValueError('identical_raw_limits: %s'%raw_limits)
        if( raw_limits[1] > raw_limits[0] ):
            self.graph_positive = True
        else:    #Swap the raw_limits around
            self.graph_positive = False
            raw_limits[0], raw_limits[1] = raw_limits[1] , raw_limits[0]

        #There must be an integer number of minor ticks for each major tick
        if not( ((raw_limits[2]/raw_limits[3]) % 1.0) * raw_limits[3] == 0 ):  
            raise ValueError('bad_tick_spacing')

        if( raw_limits[2] <= 0 or
            raw_limits[3] <= 0 or
            raw_limits[2] < raw_limits[3] or
            raw_limits[3] > abs(raw_limits[1]-raw_limits[0]) ):
            raise ValueError('bad_raw_limits:%s' % raw_limits)
        for zone in raw_zones:
            if( zone[0] > zone[1] ):    #Swap the zones so zone[1] > zone[0]
                zone[0], zone[1] = zone[1], zone[0]
            if( zone[1] < raw_limits[0] or zone[0] > raw_limits[1] ):
                raise ValueError('bad_zone:%s'%zone)
            if( zone[0] < raw_limits[0] ):
                zone[0] = raw_limits[0]
            if( zone[1] > raw_limits[1] ):
                zone[1] = raw_limits[1]

        #Stuff all of the variables into self.
        self.raw_value = raw_value
        self.raw_limits = raw_limits
        self.raw_zones = raw_zones
        self.attribute_name = attribute_name
        self.field_names = field_names
        self.file_name = file_name
        self.resolution = resolution
        self.x_length = x_length
        self.y_length = y_length
        try:
            self.degrees = raw_limits[4]
        except IndexError:
            self.degrees = 180  # Not there, assume 180 for compatibility
        self.lowlim  = 90.0 - self.degrees/2
        self.highlim = 90.0 + self.degrees/2

        #print "From", self.lowlim, 'to', self.highlim

        #Draw the gauge, each zone (color) separately
        for zone in raw_zones:
            self.draw_arch( zone, False )
        self.draw_arch( None, True )
        self.draw_ticks()
        self.draw_needle()
        self.draw_bounding_box()
        self.text(0.0, 0.2, self.attribute_name, size=10, va='bottom', ha='center')

        # The black dot. This probably should be worked in to the needle proper.
        # The nonzero [x],[y] values seem to account for the text offset of the dot.
        p = self.plot([-0.02],[-0.02],'.', color='#000000')


    def draw_arch( self, zone, just_border ):
        if( just_border ):
            start   = self.raw_limits[0]
            end     = self.raw_limits[1]
        else:
            start   = zone[0]
            end     = zone[1]
            colour  = zone[2]
           
        x_vect      = []
        y_vect      = []

        start_value = max(start, end)
        end_value   = min(start, end)
        d_theta     = 1         # Smaller for slower, more precise drawing

        luw = self.raw_limits[1]-self.raw_limits[0] # Logical units width

        # The function to adjust user units to radians, a straight linear mapping
        if self.graph_positive:
            adj = lambda a: (self.highlim-a*self.degrees/luw) * pi / 180.0
        else:
            adj = lambda a: (self.lowlim+a*self.degrees/luw) * pi / 180.0

        #Draw an arch
        theta  = start_value
        radius = 0.85
        while (theta >= end_value):
            x_vect.append( radius * math.cos( adj(theta) ) )
            y_vect.append( radius * math.sin( adj(theta) ) )
            theta -= d_theta

        theta  = end_value
        radius = 1.0
        while (theta <= start_value):
            x_vect.append( radius * math.cos( adj(theta) ) )
            y_vect.append( radius * math.sin( adj(theta) ) )
            theta += d_theta

        if( just_border ):
            #Close the loop
            x_vect.append(0.85 * math.cos( adj(start_value) ) )
            y_vect.append(0.85 * math.sin( adj(start_value) ) )

            p = self.plot(x_vect, y_vect, 'b-', color='black', linewidth=1.0)
        else:
            p = self.fill(x_vect, y_vect, colour, linewidth=0.0, alpha=0.4)


    def draw_needle( self ):
        x_vect = []
        y_vect = []

        if self.raw_value == None:
            self.text(0.0, 0.4, "N/A", size=10, va='bottom', ha='center')
        else:
            self.text(0.0, 0.4, "%.2f" % self.raw_value, size=10, va='bottom', ha='center')

            #Clamp the value to the raw_limits
            if( self.raw_value < self.raw_limits[0] ):
                self.raw_value = self.raw_limits[0]
            if( self.raw_value > self.raw_limits[1] ):
                self.raw_value = self.raw_limits[1]

            theta  = 90 - self.degrees/2
            length = 0.95
            if( self.graph_positive ):
                angle = float(self.degrees) - (self.raw_value - self.raw_limits[0]) * (float(self.degrees)/abs(self.raw_limits[1]-self.raw_limits[0]))
            else:
                angle =         (self.raw_value - self.raw_limits[0]) * (float(self.degrees)/abs(self.raw_limits[1]-self.raw_limits[0]))

            while (theta <= 90.0 + self.degrees/2):
                x_vect.append( length * math.cos((theta + angle) * (pi/180.0)) )
                y_vect.append( length * math.sin((theta + angle) * (pi/180.0)) )
                length = 0.05
                theta += self.degrees/2

            p = self.fill(x_vect, y_vect, 'b', alpha=0.4)

    def draw_ticks( self ):
        if( self.graph_positive ):
            angle = float(90.0 + self.degrees/2)
        else:
            angle = float(90.0 - self.degrees/2)
        i = 0
        j = self.raw_limits[0]

        while( i*self.raw_limits[3] + self.raw_limits[0] <= self.raw_limits[1] ):
            x_vect = []
            y_vect = []
            if( i % (self.raw_limits[2]/self.raw_limits[3]) == 0 ):
                x_pos = 1.1 * math.cos( angle * (pi/180.0))
                y_pos = 1.1 * math.sin( angle * (pi/180.0))
                if( type(self.raw_limits[2]) is types.FloatType ):
                    self.text( x_pos, y_pos, "%.2f" % j, size=10, va='center', ha='center', rotation=(angle - 90))
                else:
                    self.text( x_pos, y_pos, "%d" % int(j), size=10, va='center', ha='center', rotation=(angle - 90))
                tick_length = 0.15
                j += self.raw_limits[2]
            else:
                tick_length = 0.05
            i += 1
            x_vect.append( 1.0 * math.cos( angle * (pi/180.0)))
            x_vect.append( (1.0 - tick_length) * math.cos( angle * (pi/180.0)))
            y_vect.append( 1.0 * math.sin( angle * (pi/180.0)))
            y_vect.append( (1.0 - tick_length) * math.sin( angle * (pi/180.0)))
            p = self.plot(x_vect, y_vect, 'b-', linewidth=1, alpha=0.4, color="black")
            if( self.graph_positive ):
                angle -= self.raw_limits[3] * (float(self.degrees)/abs(self.raw_limits[1]-self.raw_limits[0]))
            else:
                angle += self.raw_limits[3] * (float(self.degrees)/abs(self.raw_limits[1]-self.raw_limits[0]))
        if( i % (self.raw_limits[2]/self.raw_limits[3]) == 0 ):
            x_pos = 1.1 * math.cos( angle * (pi/180.0))
            y_pos = 1.1 * math.sin( angle * (pi/180.0))
            if( type(self.raw_limits[2]) is types.FloatType ):
                self.text( x_pos, y_pos, "%.2f" % j, size=10, va='center', ha='center', rotation=(angle - 90))
            else:
                self.text( x_pos, y_pos, "%d" % int(j), size=10, va='center', ha='center', rotation=(angle - 90))   


    def draw_bounding_box( self ):
        x_vect = [
            self.x_length/2,
            self.x_length/2,
            -self.x_length/2,
            -self.x_length/2,
            self.x_length/2,
            ]

        y_vect = [
            -0.1,
            self.y_length,
            self.y_length,
            -0.1,
            -0.1,
            ]

        p = self.plot(x_vect, y_vect, 'r-', linewidth=0)

def make_widget( raw_value, raw_limits, raw_zones, attribute_name, field_names, file_name, resolution=72 ):
    from pylab import figure, show, savefig
  
    x_length = 8.4  # Length of the Primary axis
    y_length = 3.2  # Length of the Secondary axis
       
    fig_height = y_length
    fig_width  = x_length
    fig = figure( figsize=(fig_width, fig_height) )
    rect = [(0.0/fig_width), (0.2/fig_height), (x_length/fig_width), (y_length/fig_height)]
    gauge = Gauge( raw_value,
        raw_limits, raw_zones,
        attribute_name, field_names,
        file_name, resolution,
        x_length, y_length,
        fig, rect,
        xlim=( -0.1, x_length+0.1 ),
        ylim=( -0.4, y_length+0.1 ),
        xticks=[],
        yticks=[],
        )
           
    gauge.set_axis_off()
    fig.add_axes(gauge)

    show()
    #fig.canvas.print_figure( file_name,dpi=resolution )             
                   
   
#make_widget( -3.0, [-10.0,10.0,5,1], [[-10.0,0.0,'r'],[0.0,5.0,'y'],[5.0,10.0,'g']], "Rx MOS (24h)", ['WLL to LAS','LAS to WLL','WLL to LAS','LAS to WLL'], 'gauge.png', 100)
#make_widget(.23, [0.0,1.0,.2,.05], [[0.0,0.8,'g'],[0.8,0.95,'#ffa500'],[0.95,1,'r']], "Cache Full %", ['able','baker','charley','dog'], 'gauge.png', 100)


if __name__=='__main__':
    from pylab import figure, show
    
    raw_value  = 23.47
    raw_limits = [0.0, 100.0, 20, 5, 120] # 120-degrees
    if raw_limits[1] > raw_limits[0]:
        raw_zones  = [[0,80,'g'],[80,95,'#ffa500'],[95,100,'r']]
    else:
        raw_zones  = [[100,95,'r'],[95,80,'#ffa500'],[80,0,'g']]

    attribute_name = "Cache full %"
    field_names    = ["FILLING","FULL","OVERCOMMITTED"]

    file_name    = 'gauge.png'
    resolution   = 72

    graph_height = 2.3
    graph_width  = 4.4 

    graph_height = 6.0
    graph_width  = 9.5 

    fig_height   = graph_height
    fig_width    = graph_width

    x_length     = graph_width
    y_length     = graph_height

    fig = figure( figsize=(fig_width, fig_height) )
  
    rect = [(0.15/fig_width), -(0.15/fig_height),
            (graph_width/fig_width), (graph_height/fig_height)]

    gauge = Gauge( raw_value,
                   raw_limits, raw_zones,
                   attribute_name, field_names,
                   file_name, resolution,
                   x_length, y_length,
                   fig, rect,
                   xlim=( -0.9, graph_width+0.1 ),
                   ylim=( -0.4, graph_height+0.1 ),
                   xticks=[],
                   yticks=[],
                   )

    gauge.set_axis_off()
    fig.add_axes(gauge)

    fig.canvas.print_figure('gauge',dpi=72)
    show()
  
