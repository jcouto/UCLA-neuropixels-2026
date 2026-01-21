### Waveplayer 

Use the waveplayer in SpikeGLX to configure output channels during acquisition.

This is the complete package prepared by _Jennifer Colonell_ for running the SDK through python -- you don't need to download or install anything else. These examples are for Onebox but work also with NIdaq.

### Instructions 

1. In **SpikeGLX**, enable the command server**, start a run with OBX enabled, analog input = 1:11, AO = 0
2. Connect the ADC output (SMA on OneBox) to IO1 on the breakout board. Note that IO1 will be channel 0.
3. **Copy the wave files** (inside the _Waves folder) to the SpikeGLX\_Waves folder (don't copy the folder)
4. in __wp_soft_start.py__, edit the slot number to match your Onebox
5. Run the script -- you should see the programmed voltages on Channel 0 of the obx graphs window.
6. The example runs one waveform with multiple sine frequencies, with 2 sec pauses in between. If you want more control (like, you want to scramble the stimulus frequencies), make a waveform file for each one.


### About the files 

``wp_arbitrary_voltage.py`` outputs a random voltage on channel 0.

``wp_soft_start.py`` sends a waveform to channel 0 and triggers with software.

``wp_trial_hw_trig.py`` sends a waveform to channel 0 and triggers with hardware. Use this for instance to trigger the waveplayer with a TTL. Currently in this example we trigger the board with itself.




