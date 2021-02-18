rmax = 0.5
squeezefac = 1.0
  if isstr(loc_file)
      [tmpeloc labels Th Rd indices] = readlocs( loc_file);
  elseif isstruct(loc_file) % a locs struct
      [tmpeloc labels Th Rd indices] = readlocs( loc_file );
      % Note: Th and Rd correspond to indices channels-with-coordinates only
  else
       error('loc_file must be a EEG.locs struct or locs filename');
  end
  Th = pi/180*Th;                              % convert degrees to radians

  if isempty(plotrad) 
  plotrad = min(1.0,max(Rd)*1.02);            % default: just outside the outermost electrode location
  plotrad = max(plotrad,0.5);                 % default: plot out to the 0.5 head boundary
end                                           % don't plot channels with Rd > 1 (below head)

if isempty(intrad) 
  default_intrad = 1;     % indicator for (no) specified intrad
  intrad = min(1.0,max(Rd)*1.02);             % default: just outside the outermost electrode location
else
  default_intrad = 0;                         % indicator for (no) specified intrad
  if plotrad > intrad
     plotrad = intrad;
  end
end                                           % don't interpolate channels with Rd > 1 (below head)
if isempty(headrad)  % never set -> defaults
  if plotrad >= rmax
     headrad = rmax;  % (anatomically correct)
  else % if plotrad < rmax
     headrad = 0;    % don't plot head
     if strcmpi(VERBOSE, 'on')
       fprintf('topoplot(): not plotting cartoon head since plotrad (%5.4g) < 0.5\n',...
                                                                    plotrad);
     end
  end

elseif strcmpi(headrad,'rim') % force plotting at rim of map
  headrad = plotrad;
end
