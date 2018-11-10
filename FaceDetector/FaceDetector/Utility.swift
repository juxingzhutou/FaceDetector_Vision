//
//  utility.swift
//
//  Created by Riceberg on 2018-11-08.
//  Copyright © 2018 Riceberg. All rights reserved.
//

import Foundation

extension Double {
    func format(f: String) -> String {
        return String(format: "%\(f)f", self)
    }
}
