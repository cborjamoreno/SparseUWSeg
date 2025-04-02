    def get_best_point(self):
        """
        Get the best point using a simple approach:
        1. Calculate centroid of each mask
        2. Cluster very close centroids together
        3. Select the centroid of the largest unselected mask
        """
        if not hasattr(self, 'selected_points'):
            self.selected_points = []
        
        # Calculate centroids and areas for all unselected masks
        mask_info = []
        for mask in self.masks:
            if id(mask) not in self.selected_masks:
                segmentation = mask['segmentation']
                indices = list(zip(*segmentation.nonzero()))
                if indices:
                    centroid = np.mean(indices, axis=0)
                    area = mask['area']
                    mask_info.append({
                        'centroid': centroid,
                        'area': area,
                        'mask_id': id(mask)
                    })
        
        if not mask_info:
            return None
            
        # Sort masks by area in descending order
        mask_info.sort(key=lambda x: x['area'], reverse=True)
        
        # Simple clustering of very close centroids
        min_distance = 20  # Minimum distance between centroids to consider them different
        clustered_info = []
        
        for info in mask_info:
            centroid = info['centroid']
            # Check if this centroid is close to any already clustered centroid
            merged = False
            for cluster in clustered_info:
                dist = np.sqrt(np.sum((centroid - cluster['centroid']) ** 2))
                if dist < min_distance:
                    # Merge with existing cluster (weighted average by area)
                    total_area = cluster['area'] + info['area']
                    cluster['centroid'] = (cluster['centroid'] * cluster['area'] + 
                                         centroid * info['area']) / total_area
                    cluster['area'] = total_area
                    cluster['mask_ids'].add(info['mask_id'])
                    merged = True
                    break
            
            if not merged:
                # Create new cluster
                clustered_info.append({
                    'centroid': centroid,
                    'area': info['area'],
                    'mask_ids': {info['mask_id']}
                })
        
        # Find the first cluster that's far enough from all selected points
        min_selection_distance = 50  # Minimum distance from previously selected points
        
        for cluster in clustered_info:
            centroid = cluster['centroid']
            # Check distance to all previously selected points
            too_close = False
            for selected_point in self.selected_points:
                dist = np.sqrt(np.sum((centroid - selected_point) ** 2))
                if dist < min_selection_distance:
                    too_close = True
                    break
            
            if not too_close:
                # Found a good point, update selected masks and points
                self.selected_masks.update(cluster['mask_ids'])
                self.selected_points.append(centroid)
                return tuple(centroid.astype(int))
        
        # If we couldn't find a point far enough from selected points
        return None 