a=prnist([0:9], [1:10]);
%nisttrain_cell, cell datafile with 40 objects in 2 crisp classes: [20  20]
%show(a)
% add rows and columns to create square figure (aspect ratio 1)
b = im_box(a,[],1); %figure; show(b)
% resample by 16 x 16 pixels
c = im_resize(b,[16,16]); %figure; show(c)

matImages = c*data2im()
matLabels = getnlab(c)'
%s2 = getfeatsize(a)

% compute means
%d = im_mean(c);
% convert to PRTools dataset and display
%x = prdataset(d,[],'featlab',char('mean-x','mean-y'));
%scatterd(x,'legend');