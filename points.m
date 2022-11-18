indir = ('data/Tripod/');
outdir = ('head_tail_coords');
if ~exist(outdir)
    mkdir(outdir);
end
d = dir([indir filesep '*.jpg']);
for i = 1:2
    im = imread([indir filesep d(i).name]);
    figure;
    imagesc(im);
    [x,y] = ginput(2);
    csvwrite([outdir filesep d(i).name(1:end-3) 'csv'], ceil([x,y]));
    close all;
end
