% read FLIR jpeg files - OF

filename = 'FLIR1938.jpg';
v = FlirMovieReader(filename);  
v.unit = 'temperatureFactory';    
while ~isDone(v)                                                    
    [frame, metadata] = step(v);                                    
end