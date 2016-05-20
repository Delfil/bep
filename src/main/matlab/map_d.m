function M = map_d(P, verbose)
% MAP maps an m by 2 set of 2d points P to a matrix containing the index of
% that point.

if ~exist('verbose', 'var')
    verbose = false;
end

%M, the raster containing P indices
% M = zeros(ceil(sqrt(size(P,1))));
M = [0];

%outer borders of the raster
ma = max(P,1);
mi = min(P,1);

% devide up the x and y values of the raster
x = linspace(mi(1),ma(1),size(M,2));
y = linspace(mi(2),ma(2),size(M,1));

% start with some point in P
points = randperm(size(P,1));
point_ind = 1;
same = 0;
iters = 1;

while point_ind <= numel(points)
    
    %     take a point
    point = P(points(point_ind),:);
    
    %     the lowest index in M it could have is:
    index = [1, 1];
    
    %     get the actual index
    %     The index will be the the indices of the smallest entries in x and y
    %     which are larger than the x and y coordinates of the point. In other
    %     words, a point is represented by the upper right corner of a
    %     rectangle it is in. The size of that rectangele is at most one over
    %     the amount of points times the rectangular area of the point cloud.
    for ind = 2:max(numel(x),numel(y))
        if ind <= numel(x) && x(ind) < point(1)
            index(2) = ind;
        end
%         elseif ind > numel(y) || y(ind) > point(2)
%             break;
%         end
        if ind <= numel(y) && y(ind) < point(2)
            %trying to conserve the original picture instead of mirroring
            index(1) = ind;
            %index(1) = numel(y)+1 - ind;
        end
    end
    
    %     if the designated index in M is not empty
    if M(index(1),index(2)) ~= 0
        %         get the other point
        point2 = P(M(index(1),index(2)),:);
        %         determine the distance between the points
        dist = abs(point2 - point);
        %         get the middle between them
        mid = .5*dist + min(point, point2);
        
        
        %         if the distance is zero
        if dist(1) == dist(2) && dist(1) == 0
%             yet to be determined
            disp('Todo: fix this!');
            
        else
            
            if dist(1) > dist(2)
                %if distance is largest in the x direction
                
                %add extra granularity to x
                extra = zeros(numel(y),1);
                
                %check that the matrix is still consistent
                for i = 1:numel(y);
                    if M(i,index(2)) ~= 0 && P(M(i,index(2)),1) > mid(1)
                        extra(i) = M(i, index(2));
                        M(i, index(2)) = 0;
                    end
                end
                
                if index(2) < numel(x);
                    x = [x(1:index(2)), mid(1), x(index(2)+1:end)];
                    M = [M(:,1:index(2)), extra, M(:,index(2)+1:end)];
                else
                    x = [x, mid(1)];
                    M = [M, extra];
                end
            else
                %if the distance is largest in the y direction or if they
                %are equal
                
                %add extra granularity to y
                extra = zeros(1, numel(x));
                
                %check that 
                for i = 1:numel(x);
                    if M(index(1),i) ~= 0 && P(M(index(1),i),2) > mid(2)
                        extra(i) = M(index(1),i);
                        M(index(1),i) = 0;
                    end
                end
                
                if index(1) < numel(y);
                    y = [y(1:index(1)), mid(2), y(index(1)+1:end)];
                    M = [M(1:index(1),:); extra; M(index(1)+1:end,:)];
                else
                    y = [y, mid(2)];
                    M = [M; extra];
                end
            end
            
        end
        %     if the designated index in M is empty
    else
        %assign value to the right point
        M(index(1),index(2)) = points(point_ind);
        point_ind = point_ind + 1;
    end
    
    %show progress
    iters = iters +1;
    if mod(iters, 1000) == 0 && verbose
        disp(sprintf('Iteration\t%s:\t%s%% done.', num2str(iters-1), num2str(floor(100*point_ind/numel(points)))))
    end
end
if verbose
    disp(sprintf('Iteration\t%s:\t%s%% done.', num2str(iters-1), num2str(floor(100*(point_ind-1)/numel(points)))))
end
end
