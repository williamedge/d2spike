![](assets/despike_eg_small.gif)

Code to de-spike time series data based on the Goring-Nikora (2002) and Wahl (2003) method. 

The package is specifically intended for de-spiking data from acoustic profiler instruments (e.g. ADCPs) which are 2D. It has been tested on heavily spiked ADCP data from the pelagic bottom boundary layer. The key improvements over the basic Goring-Nikora implementation are:
 - Use of correlation data (supplied by instrument) to pre-emptively remove likely poor data before lowpass filtering.
 - 2D Gaussian filtering to determine the mean (background) velocity. This is a fundamental and potentially difficult part of the method. Incorrect determination of the time series mean will violate the methods assumptions and cause erroneous spike detection.
 - 2D interpolation after the de-spiking algorithm. This is optional and may not be a good idea depending on the intended use of the data.

