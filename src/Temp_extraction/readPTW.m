% This function reads .ptw files from dataPath specified by user 
%
% ---------------------------Input Arguments------------------------------%
% dataPath: string, specifies path from which files are to be imported
%
% subfolders: 0 or 1, specifies if files are organized in subfolders or are
% all in one folder as specified by dataPath. 1 means sub-folder
% architecture, 0 direct architecture. (Default 0)
%
% ---------------------------Output Arguments-----------------------------%
% frames: a 256*320*N double matrix containing temperature matrix for each
% imported file
%
% files: a struct with information about imported files. frames is indexed
% as in files. 
%
%-------------------------------------------------------------------------%

function [frames, files] = readPTW(dataPath,subfolders)

narginchk(1,2);                                                             %dataPath is obligatory while subfolders is set to 0 by default if not set by user

if ~exist('subfolders', 'var')
    subfolders = 0;                                                         %set subfolders to 0 if not specified by user
end

if subfolders == 0
    files = dir([dataPath '/*.ptw']);                                       
    frames  = zeros(240,320,length(files));
    
    for index_file = 1:length(files)
        
        v = FlirMovieReader([dataPath '\' files(index_file).name]);         % open the file
        v.unit = 'temperatureFactory';                                      % set the desired unit
        while ~isDone(v)                                                    % loops for all frames in the file
            [frame, metadata] = step(v);                                    % read the next frame
        end
        
        frames(:,:,index_file) = frame;
        
        delete(v);                                                          % free the file object
    end
    
    for i = 1:length(files)
        [files(i).folder] = [dataPath];
    end
    
    if isempty(frames)
        warning('Frames returned empty. Check that path is valid. If images are in subfolders set second input argument to 1')
    end
elseif subfolders == 1
    folders = dir([dataPath]);
    
    folders(1) = [];
    folders(1) = [];
    
    for index_folder = 1:length(folders)
        foldername = folders(index_folder).name;
        
        if index_folder == 1
            files = dir([dataPath '\' foldername '/*.ptw']);
            for i = 1:length(files)
                [files(i).folder] = [dataPath '\' foldername];
            end
        else
            file = dir([dataPath '\' foldername '/*.ptw']);
            if length(file) ~= 0
                for i = 1:length(file)
                    [file(i).folder] = [dataPath '\' foldername];
                end
                files = [files;file];
            end
        end
    end
    
    frames  = zeros(240,320,length(files));
    
    for index_file = 1:length(files)
        
        filename = files(index_file).name;
        
        v = FlirMovieReader([files(index_file).folder '\' filename]);       % open the file
        v.unit = 'temperatureFactory';                                      % set the desired unit
        while ~isDone(v)                                                    % loops for all frames in the file
            [frame, metadata] = step(v);                                    % read the next frame
        end
        
        frames(:,:,index_file) = frame;
        
        delete(v);                                                          % free the file object
    end
    
    if isempty(frames)
        warning('Frames returned empty. Check that path is valid. If images are not in subfolders set second input argument to 0')
    end
else
    error('Second input argument must be either 1 or 0');
end

return