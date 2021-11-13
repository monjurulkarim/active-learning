''' Pseudocode'''
''' Input '''
    # Class_id     # Classes of objects
    # Score        # detection scores of objects
    # area         # area of detected objects
    # vertices     # vertices of masks of detected objects
    # relative_distance # ralative distance between object to object
    # distance_type # closest, medium far, far
    # relation_type # primary, secondary, tartiary
    #
    # for objects in class_id:
    #     if score is low:
    #         find the related object
    #         if related object == True:
    #             find the distance type based on area
    #             if distance_type is same as the primary object:
    #                  find the relation type
    #                  find the relative distance based on relation type
    #                  if the relative distance is within the range:
    #                      keep the object
    #                  else:
    #                      suppress the object
    #                  end if
    #             end if
    #         end if
    #     end if
    # end for

'''
The below information is the area of the polygons. There are three groups
of objects. Group_3 is the closest, group_2 is relatively far and group_3 is
very far.
object_[group no] = area
'''
# TO-Do:
# calculate the following areas from Ground Truth data, by reading from JSON file
# Find a machine learning/ statistical way to calculating the area from GT
pier_1 = 1045
pier_2 = 4632
pier_3 = 8219

pier_cap_1 = 467
pier_cap_2 = 17492
pier_cap_3 = 34500

truss_1 = 24867
truss_2 = 26500
truss_3 = 28145

barrier_1 = 3462
barrier_2 = 7681
barrier_3 = 11900

slab_1 = 50633
slab_2 = 56150
slab_3 = 61667

joint_1 = 773
joint_2 = 1077
joint_3 = 1380

'''
Distance between object to object within the same group. For example, pier_1
to pier_cap_1 should be calculated.
'''
pier_pier_cap_11 = 20
pier_pier_cap_22 = 40
pier_pier_cap_33 = 60
