
% load data here
load('data.mat')

%% define co-active frames
pks = 1;
col = find(sum(spike,1)>pks);

%% creat labels
ncond = 5; % 5 conditions
expt_time = 6000; % 6000 frames per condition

% create condition labels
expt_indx = [];
for i = 1:ncond
    expt_indx(end+1:end+expt_time) = i;
end

% keep only active frames
expt_indx = expt_indx(col);

%% tsne
% adjust perplexity to make sure error residual changes make sense
emCoord = tsne(spike(:,col)',[],2,30,1000);

%% density map
maxVal = max(emCoord(:));
maxVal = round(maxVal * 1.1);

% these are parameters to adjust
sigma = maxVal/35; % ADJUST: default /40, larger number gives more fragmented result
numPoints = 501;
rangeVals = [-maxVal maxVal];

% generate plot with all data
[xx,densAll] = findPointDensity(emCoord,sigma,numPoints,rangeVals);
maxDensity = max(densAll(:));

figure;set(gcf,'color','w')
for i = 1:5
    subplot(1,6,i)
    [~,dens] = findPointDensity(emCoord(expt_indx==i,:),sigma,numPoints,rangeVals);
    imagesc(xx,xx,dens)
    axis equal tight off xy
    caxis([0 maxDensity*0.8])
    colormap(parula)
end

subplot(1,6,6);
imagesc(xx,xx,densAll)
axis equal tight off xy
caxis([0 maxDensity*0.8])
colormap(parula)

% segmentation
im = densAll;
map_thresh = 0.5;

% smooth
fgauss = fspecial('gaussian',3,1);

% internal marker
local_max = round(FastPeakFind(im,0,fgauss));
int_marker = false(size(im));
int_marker(sub2ind(size(im),local_max(2:2:end),local_max(1:2:end))) = true;
int_marker = imdilate(int_marker,strel('disk',3));
%     intm_dist = imcomplement(bwdist(~int_marker));
intm_dist = (max(im(:))-im).*(~int_marker);

% watershed
seg_im = watershed(intm_dist);
seg_im_trans = seg_im';

% zero-set boundaries
seg_bound = seg_im==0;
numClass = length(unique(seg_im(:)))-1;

vdata = emCoord;
vdata = round((vdata/maxVal*numPoints+numPoints)/2);
vdata(vdata<=0) = 1;
vdata(vdata>=numPoints) = numPoints;
segIndx = seg_im_trans(sub2ind(size(im),vdata(:,1),vdata(:,2)));

%% plot segmentation
figure;
set(gcf,'color','w','position',[1975 438 1114 500],'PaperPositionMode','auto');
subplot(1,2,1);
imagesc(xx,xx,(double(seg_im).*double(im>quantile(im(:),map_thresh)))==0);
colormap(gray);
hold on;
gscatter(emCoord(:,1),emCoord(:,2),expt_indx);
axis equal tight off xy
title('watershed segmentation');
subplot(1,2,2);
imagesc(xx,xx,(double(seg_im).*double(im>quantile(im(:),map_thresh)))==0);
colormap(gray);
axis equal tight off xy
title('watershed regions');
region_cent = regionprops(double(seg_im).*double(im>quantile(im(:),...
    map_thresh)),'Centroid');
for i = 1:length(region_cent)
    tmpCoord = [region_cent(i).Centroid(1),region_cent(i).Centroid(2)];
    if isnan(tmpCoord(1))
        continue;
    end
    tmpNum = seg_im(round(tmpCoord(2)),round(tmpCoord(1)));
    if tmpNum~=0
        h = text(2*maxVal*(tmpCoord(1)-numPoints/2)/numPoints,...
            2*maxVal*(tmpCoord(2)-numPoints/2)/numPoints,num2str(tmpNum));
        set(h,'color','y','fontsize',10);
    end
end

%     saveas(gcf,[figpath filename{n} '_segmentation.fig']);

% plot overlay segmentation
figure;set(gcf,'color','w')
imagesc(xx,xx,im.*(~seg_bound));
caxis([0 max(im(:))*0.8]);colormap(jet);colorbar;
axis equal tight off xy
title('watershed segmentation');

%% tsne shuffle
nshuff = 100;
ncell = size(spike,1);
seg_im_shuff = cell(1,nshuff);
segIndx_shuff = cell(nshuff,1);
emCoord_shuff = cell(nshuff,1);
for k = 1:nshuff
    
    fprintf('shuffle #%u\n',k);
    
    spike_shuff = spike(:,col);
    for ii = 1:length(col)
        indx = randperm(ncell);
        spike_shuff(:,ii) = spike_shuff(indx,ii);
    end
    
    emCoord_shuff{k} = tsne(spike_shuff',[],2,30,1000);
    
end

%% shuffle segmentation
for k = 1:nshuff
    
    maxVal = max(emCoord_shuff{k}(:));
    maxVal = round(maxVal * 1.1);
%     maxVal = 6;
    
    % these are parameters to adjust
    sigma = maxVal/35; % ADJUST: default /40, larger number gives more fragmented result
    numPoints = 501;
    rangeVals = [-maxVal maxVal];
    
    % generate plot with all data
    [xx,densAll] = findPointDensity(emCoord_shuff{k},sigma,numPoints,rangeVals);
    
    % segmentation
    im = densAll;
    map_thresh = 0.5;
    
    % smooth
    fgauss = fspecial('gaussian',3,1);
    
    % internal marker
    local_max = round(FastPeakFind(im,0,fgauss));
    int_marker = false(size(im));
    int_marker(sub2ind(size(im),local_max(2:2:end),local_max(1:2:end))) = true;
    int_marker = imdilate(int_marker,strel('disk',3));
    %     intm_dist = imcomplement(bwdist(~int_marker));
    intm_dist = (max(im(:))-im).*(~int_marker);
    
    % watershed
    seg_im_shuff{k} = watershed(intm_dist)';
    
    vdata = emCoord_shuff{k};
    vdata = round((vdata/maxVal*numPoints+numPoints)/2);
    vdata(vdata<=0) = 1;
    vdata(vdata>=numPoints) = numPoints;
    segIndx_shuff{k} = seg_im_shuff{k}(sub2ind(size(im),vdata(:,1),vdata(:,2)));
    
end


%% tsne number of states
num_state_tsne = zeros(ncond,1);
for n = 1:ncond
    num_state_tsne(n) = length(unique(segIndx(expt_indx==n)));
end

% shuffle
num_state_tsne_shuff = zeros(ncond,nshuff);
for n = 1:ncond
    for k = 1:nshuff
        num_state_tsne_shuff(n,k) = length(unique(segIndx_shuff{k}(expt_indx==n)));
    end
end


%% save results
save('embedding_state_results.mat','col','emCoord','expt_indx','densAll','seg_im',...
    'segIndx','segIndx_shuff','num_state_tsne','num_state_tsne_shuff','-v7.3');



